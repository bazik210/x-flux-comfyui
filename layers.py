import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

import comfy.model_management

from .xflux.src.flux.math import attention, rope
from .xflux.src.flux.modules.layers import LoRALinearLayer

from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from .attention_processor import IPAFluxAttnProcessor2_0

from torch.nn import functional as F
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class DoubleStreamBlockIPA(nn.Module):
    def __init__(self, original_block: DoubleStreamBlock, ip_adapter: list[IPAFluxAttnProcessor2_0], image_emb):
        super().__init__()

        self.num_heads = original_block.num_heads
        self.hidden_size = original_block.hidden_size
        self.img_mod = original_block.img_mod
        self.img_norm1 = original_block.img_norm1
        self.img_attn = original_block.img_attn

        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp

        self.txt_mod = original_block.txt_mod
        self.txt_norm1 = original_block.txt_norm1
        self.txt_attn = original_block.txt_attn

        self.txt_norm2 = original_block.txt_norm2
        self.txt_mlp = original_block.txt_mlp
        self.flipped_img_txt = original_block.flipped_img_txt

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = comfy.model_management.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)
    
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, t: Tensor, attn_mask=None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if self.flipped_img_txt:
             # run actual attention
            q = torch.cat((img_q, txt_q), dim=2)
            k = torch.cat((img_k, txt_k), dim=2)
            v = torch.cat((img_v, txt_v), dim=2)
            attn_out = attention(q, k, v, pe=pe, mask=attn_mask)
            img_attn, txt_attn = attn_out[:, :img.shape[1]], attn_out[:, img.shape[1]:]    
        else:
            # run actual attention
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)
            attn_out = attention(q, k, v, pe=pe, mask=attn_mask)
            txt_attn, img_attn = attn_out[:, :txt.shape[1]], attn_out[:, txt.shape[1]:]

        for adapter, image in zip(self.ip_adapter, self.image_emb):
            # this does a separate attention for each adapter
            ip_hidden_states = adapter(self.num_heads, img_q, image, t)
            if ip_hidden_states is not None:
                ip_hidden_states = ip_hidden_states[0].to(self.device)
                #ip_hidden_states = ip_hidden_states.to(self.device)
                img_attn = img_attn + ip_hidden_states

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt

class SingleStreamBlockIPA(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, original_block: SingleStreamBlock, ip_adapter: list[IPAFluxAttnProcessor2_0], image_emb):
        super().__init__()
        self.hidden_dim = original_block.hidden_size
        self.num_heads = original_block.num_heads
        self.scale = original_block.scale

        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        # qkv and mlp_in
        self.linear1 = original_block.linear1
        # proj and mlp_out
        self.linear2 = original_block.linear2

        self.norm = original_block.norm

        self.hidden_size = original_block.hidden_size
        self.pre_norm = original_block.pre_norm

        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = comfy.model_management.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, t:Tensor, attn_mask=None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)

        for adapter, image in zip(self.ip_adapter, self.image_emb):
            # this does a separate attention for each adapter
            # maybe we want a single joint attention call for all adapters?
            ip_hidden_states = adapter(self.num_heads, q, image, t)
            if ip_hidden_states is not None:
                ip_hidden_states = ip_hidden_states[0].to(self.device)
                #ip_hidden_states = ip_hidden_states.to(self.device)
                attn = attn + ip_hidden_states

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

class DoubleStreamBlockLorasMixerProcessor(nn.Module):
    def __init__(self,):
        super().__init__()
        self.qkv_lora1 = []
        self.proj_lora1 = []
        self.qkv_lora2 = []
        self.proj_lora2 = []
        self.lora_weight = []
        self.names = []
    def add_lora(self, processor):
        if isinstance(processor, DoubleStreamBlockLorasMixerProcessor):
            self.qkv_lora1+=processor.qkv_lora1
            self.qkv_lora2+=processor.qkv_lora2
            self.proj_lora1+=processor.proj_lora1
            self.proj_lora2+=processor.proj_lora2
            self.lora_weight+=processor.lora_weight
        else:
            if hasattr(processor, "qkv_lora1"):
                self.qkv_lora1.append(processor.qkv_lora1)
            if hasattr(processor, "proj_lora1"):
                self.proj_lora1.append(processor.proj_lora1)
            if hasattr(processor, "qkv_lora2"):
                self.qkv_lora2.append(processor.qkv_lora2)
            if hasattr(processor, "proj_lora2"):
                self.proj_lora2.append(processor.proj_lora2)
            if hasattr(processor, "lora_weight"):
                self.lora_weight.append(processor.lora_weight)
    def get_loras(self):
        return (
            self.qkv_lora1, self.qkv_lora2, 
            self.proj_lora1, self.proj_lora2,
            self.lora_weight
        )
    def set_loras(self, qkv1s, qkv2s, proj1s, proj2s, w8s):
        for el in qkv1s:
            self.qkv_lora1.append(el)
        for el in qkv2s:
            self.qkv_lora2.append(el)
        for el in proj1s:
            self.proj_lora1.append(el)
        for el in proj2s:
            self.proj_lora2.append(el)
        for el in w8s:
            self.lora_weight.append(el)
        
    def add_shift(self, layer, origin, inputs, gating = 1.0):
        #shift = torch.zeros_like(origin)
        count = len(layer)
        for i in range(count):
            origin += layer[i](inputs)*self.lora_weight[i]*gating
        
    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        #img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_qkv = attn.img_attn.qkv(img_modulated)
        #print(self.qkv_lora1)
        self.add_shift(self.qkv_lora1, img_qkv, img_modulated)
            
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        
        #txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        self.add_shift(self.qkv_lora2, txt_qkv, txt_modulated)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        #img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) 
        self.add_shift(self.proj_lora1, img, img_attn, img_mod1.gate)
        
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        
        # calculate the txt bloks
        #txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) 
        self.add_shift(self.proj_lora2, txt, txt_attn, txt_mod1.gate)
        
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlockProcessor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        self.__call__(attn, img, txt, vec, pe, **attention_kwargs)


class IPProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim, ip_hidden_states=None, ip_scale=None, text_scale=None):
        super().__init__()
        device = comfy.model_management.get_torch_device()
        self.ip_hidden_states = ip_hidden_states
        self.ip_scale = ip_scale
        self.text_scale = text_scale
        self.in_hidden_states_neg = None
        self.in_hidden_states_pos = ip_hidden_states
        # Ensure context_dim matches the dimension of ip_hidden_states
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        if text_scale is None:
            self.text_scale=1.0
        if self.text_scale is None:
            self.text_scale=1.0
        if self.ip_scale is None:
            self.ip_scale=1.0
        if self.text_scale == 0:
            self.text_scale = 0.0001
        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True).to(device)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True).to(device)
        
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)
        
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def forward(self, img_q, attn):
        #img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # IP-adapter processing
        device = comfy.model_management.get_torch_device()
        ip_query = img_q.to(device)  # latent sample query
        ip_hidden_states = self.ip_hidden_states.to(device)
        ip_key = self.ip_adapter_double_stream_k_proj(self.ip_hidden_states)
        ip_value = self.ip_adapter_double_stream_v_proj(self.ip_hidden_states)
        
        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads)
        #img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query, 
            ip_key, 
            ip_value, 
            dropout_p=0.0, 
            is_causal=False
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads)
        return ip_attention*self.ip_scale

class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        dtype = self.proj.weight.dtype
        embeds = image_embeds.to(dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class DoubleStreamMixerProcessor(DoubleStreamBlockLorasMixerProcessor):
    def __init__(self):
        super().__init__()
        self.ip_adapters = nn.ModuleList()
        
    def add_ipadapter(self, ip_adapter):
        self.ip_adapters.append(ip_adapter)

    def get_ip_adapters(self):
        return self.ip_adapters
    
    def set_ip_adapters(self, ip_adapters):
        self.ip_adapters = ip_adapters
    
    def shift_ip(self, img_qkv, attn, x):
        for block in self.ip_adapters:
            #x = x*block.text_scale
            x += torch.mean(block(img_qkv, attn), dim=0, keepdim=True)
        return x
    
    def scale_txt(self, txt):
        for block in self.ip_adapters:
            txt = txt * block.text_scale
        return txt
    
    def add_lora(self, processor):
        if isinstance(processor, DoubleStreamBlockLorasMixerProcessor):
            self.qkv_lora1 += processor.qkv_lora1
            self.qkv_lora2 += processor.qkv_lora2
            self.proj_lora1 += processor.proj_lora1
            self.proj_lora2 += processor.proj_lora2
            self.lora_weight += processor.lora_weight
        elif isinstance(processor, DoubleStreamMixerProcessor):
            self.qkv_lora1 += processor.qkv_lora1
            self.qkv_lora2 += processor.qkv_lora2
            self.proj_lora1 += processor.proj_lora1
            self.proj_lora2 += processor.proj_lora2
            self.lora_weight += processor.lora_weight
        else:
            if hasattr(processor, "qkv_lora1"):
                self.qkv_lora1.append(processor.qkv_lora1)
            if hasattr(processor, "proj_lora1"):
                self.proj_lora1.append(processor.proj_lora1)
            if hasattr(processor, "qkv_lora2"):
                self.qkv_lora2.append(processor.qkv_lora2)
            if hasattr(processor, "proj_lora2"):
                self.proj_lora2.append(processor.proj_lora2)
            if hasattr(processor, "lora_weight"):
                self.lora_weight.append(processor.lora_weight)

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        #img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_qkv = attn.img_attn.qkv(img_modulated)
        #print(self.qkv_lora1)
        self.add_shift(self.qkv_lora1, img_qkv, img_modulated)
            
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        
        #txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        self.add_shift(self.qkv_lora2, txt_qkv, txt_modulated)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        #img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        self.add_shift(self.proj_lora1, img, img_attn, img_mod1.gate)        
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        
        img = self.shift_ip(img_q, attn, img)
        # calculate the txt bloks
        #txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) 
        
        
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        #txt = self.scale_txt(txt)
        self.add_shift(self.proj_lora2, txt, txt_attn, txt_mod1.gate)

        return img, txt
