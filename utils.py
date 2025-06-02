from comfy.ldm.flux.layers import DoubleStreamBlock as DSBold
import copy
import torch
import numpy as np
import logging
from types import MethodType
from comfy.ldm.flux.layers import timestep_embedding
from .xflux.src.flux.modules.layers import DoubleStreamBlock as DSBnew
from .layers import (DoubleStreamBlockLoraProcessor,
                     DoubleStreamBlockProcessor,
                     DoubleStreamBlockLorasMixerProcessor,
                     DoubleStreamMixerProcessor,
                     DoubleStreamBlockIPA,
                     SingleStreamBlockIPA)

from comfy.utils import get_attr, set_attr
from torch import Tensor, nn

def FluxUpdateModules(bi, pbar=None, ip_attn_procs=None, image_emb=None, is_patched=False):
    """
    Apply IP-Adapter attention processors to the Flux model.
    
    Args:
        bi: model object to patch methods and attributes.
        ip_attn_procs: Dictionary of IP-Adapter attention processors.
        image_emb: Image embeddings for attention.
        is_patched: Boolean indicating if model is already patched.
    """
    flux_model = bi.model
    bi.add_object_patch(f"diffusion_model.forward_orig", MethodType(forward_orig_ipa, flux_model.diffusion_model))
    
    # Patch double blocks
    for i, original in enumerate(flux_model.diffusion_model.double_blocks):
        patch_name = f"double_blocks.{i}"
        if patch_name not in ip_attn_procs:
                logging.debug(f"Skipping {patch_name} as no IP-Adapter processor found")
                continue
        maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
        # if there's already a patch there, collect its adapters and replace it        
        procs = [ip_attn_procs[patch_name]]
        embs = [image_emb]
        if isinstance(maybe_patched_layer, DoubleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        # initial ipa models with image embeddings
        new_layer = DoubleStreamBlockIPA(original, procs, embs)
        # TODO: maybe there's a different patching method that will automatically chain patches?
        # for example, ComfyUI internally uses model.add_patches to add loras
        bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
        
def forward_orig_ipa(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor|None = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                else:
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out
            out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap,"transformer_options": transformer_options})
            txt = out["txt"]
            img = out["img"]
        else:
            if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)
    
    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                if isinstance(block, SingleStreamBlockIPA): # ipadaper
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                else:
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap, "transformer_options": transformer_options})
            img = out["img"]
        else:
            if isinstance(block, SingleStreamBlockIPA): # ipadaper
                img = block(img, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img

def CopyDSB(old_block):
    hidden_size = old_block.hidden_size
    num_heads = old_block.num_heads
    mlp_hidden_dim = old_block.img_mlp[0].out_features
    mlp_ratio = mlp_hidden_dim / hidden_size

    # Определяем qkv_bias по наличию .bias у qkv
    qkv_bias = old_block.img_attn.qkv.bias is not None

    # Получаем dtype и device с первого параметра
    param = next(old_block.parameters())
    dtype = param.dtype
    device = param.device

    # Клонируем блок
    new_block = DSBnew(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias
    ).to(dtype=dtype, device=device)

    new_block.load_state_dict(old_block.state_dict())
    new_block.set_processor(DoubleStreamBlockProcessor())

    return new_block

def copy_model(orig, new):
    new = copy.copy(new)
    new.model = copy.copy(orig.model)
    new.model.diffusion_model = copy.copy(orig.model.diffusion_model)
    new.model.diffusion_model.double_blocks = copy.deepcopy(orig.model.diffusion_model.double_blocks)
    count = len(new.model.diffusion_model.double_blocks)
    for i in range(count):
        new.model.diffusion_model.double_blocks[i] = copy.copy(orig.model.diffusion_model.double_blocks[i])
        new.model.diffusion_model.double_blocks[i].load_state_dict(orig.model.diffusion_model.double_blocks[0].state_dict())
"""
class PbarWrapper:
    def __init__(self):
        self.count = 1
        self.weights = []
        self.counts = []
        self.w8ts = []
        self.rn = 0
        self.rnf = 0.0
    def add(self, count, weight):
        self.weights.append(weight)
        self.counts.append(count)
        wa = np.array(self.weights)
        wa = wa/np.sum(wa)
        ca = np.array(self.counts)
        ml = np.multiply(ca, wa)
        cas = np.sum(ml)
        self.count=int(cas)
        self.w8ts = wa.tolist()
    def start(self):
        self.rnf = 0.0
        self.rn = 0
    def __call__(self):
        self.rn+=1
        return 1
"""



def is_model_pathched(model):
    def test(mod):
        if isinstance(mod, DSBnew):
            return True
        else:
            for p in mod.children():
                if test(p):
                    return True
        return False
    result = test(model)
    return result



def attn_processors(model_flux):
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, procs):

        if hasattr(module, "set_processor"):
            procs[f"{name}.processor"] = module.processor
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, procs)

        return procs

    for name, module in model_flux.named_children():
        fn_recursive_add_processors(name, module, processors)
    return processors
def merge_loras(lora1, lora2):
    new_block = DoubleStreamMixerProcessor()
    if isinstance(lora1, DoubleStreamMixerProcessor):
        new_block.set_loras(*lora1.get_loras())
        new_block.set_ip_adapters(lora1.get_ip_adapters())
    elif isinstance(lora1, DoubleStreamBlockLoraProcessor):
        new_block.add_lora(lora1)
    else:
        pass
    if isinstance(lora2, DoubleStreamMixerProcessor):
        new_block.set_loras(*lora2.get_loras())
        new_block.set_ip_adapters(lora2.get_ip_adapters())
    elif isinstance(lora2, DoubleStreamBlockLoraProcessor):
        new_block.add_lora(lora2)
    else:
        pass
    return new_block

def set_attn_processor(model_flux, processor):
    r"""
    Sets the attention processor to use to compute attention.

    Parameters:
        processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            for **all** `Attention` layers.

            If `processor` is a dict, the key needs to define the path to the corresponding cross attention
            processor. This is strongly recommended when setting trainable attention processors.

    """
    count = len(attn_processors(model_flux).keys())
    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
            f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        )

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            if isinstance(module.get_processor(), DoubleStreamBlockLorasMixerProcessor):
                block = copy.copy(module.get_processor())
                module.set_processor(copy.deepcopy(module.get_processor()))
                new_block = DoubleStreamBlockLorasMixerProcessor()
                #q1, q2, p1, p2, w1 = block.get_loras()
                new_block.set_loras(*block.get_loras())
                if not isinstance(processor, dict):
                    new_block.add_lora(processor)
                else:

                    new_block.add_lora(processor.pop(f"{name}.processor"))
                module.set_processor(new_block)
                #block = set_attr(module, "", new_block)
            elif isinstance(module.get_processor(), DoubleStreamBlockLoraProcessor):
                block = DoubleStreamBlockLorasMixerProcessor()
                block.add_lora(copy.copy(module.get_processor()))
                if not isinstance(processor, dict):
                    block.add_lora(processor)
                else:
                    block.add_lora(processor.pop(f"{name}.processor"))
                module.set_processor(block)
            else:
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model_flux.named_children():
        fn_recursive_attn_processor(name, module, processor)

class LATENT_PROCESSOR_COMFY:
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors =[
                    [-0.0404,  0.0159,  0.0609],
                    [ 0.0043,  0.0298,  0.0850],
                    [ 0.0328, -0.0749, -0.0503],
                    [-0.0245,  0.0085,  0.0549],
                    [ 0.0966,  0.0894,  0.0530],
                    [ 0.0035,  0.0399,  0.0123],
                    [ 0.0583,  0.1184,  0.1262],
                    [-0.0191, -0.0206, -0.0306],
                    [-0.0324,  0.0055,  0.1001],
                    [ 0.0955,  0.0659, -0.0545],
                    [-0.0504,  0.0231, -0.0013],
                    [ 0.0500, -0.0008, -0.0088],
                    [ 0.0982,  0.0941,  0.0976],
                    [-0.1233, -0.0280, -0.0897],
                    [-0.0005, -0.0530, -0.0020],
                    [-0.1273, -0.0932, -0.0680]
                ]
    def __call__(self, x):
        return (x / self.scale_factor) + self.shift_factor
    def go_back(self, x):
        return (x - self.shift_factor) * self.scale_factor



def check_is_comfy_lora(sd):
    for k in sd:
        if "lora_down" in k or "lora_up" in k:
            return True
    return False

def comfy_to_xlabs_lora(sd):
    sd_out = {}
    for k in sd:
        if "diffusion_model" in k:
            new_k =  (k
                    .replace(".lora_down.weight", ".down.weight")
                    .replace(".lora_up.weight", ".up.weight")
                    .replace(".img_attn.proj.", ".processor.proj_lora1.")
                    .replace(".txt_attn.proj.", ".processor.proj_lora2.")
                    .replace(".img_attn.qkv.", ".processor.qkv_lora1.")
                    .replace(".txt_attn.qkv.", ".processor.qkv_lora2."))
            new_k = new_k[len("diffusion_model."):]
        else:
            new_k=k
        sd_out[new_k] = sd[k]
    return sd_out

def LinearStrengthModel(start, finish, size):
    return [
        (start + (finish - start) * (i / (size - 1))) for i in range(size)
        ]
def FirstHalfStrengthModel(start, finish, size):
    sizehalf = size//2
    arr = [
        (start + (finish - start) * (i / (sizehalf - 1))) for i in range(sizehalf)
        ]
    return arr+[finish]*(size-sizehalf)
def SecondHalfStrengthModel(start, finish, size):
    sizehalf = size//2
    arr = [
        (start + (finish - start) * (i / (sizehalf - 1))) for i in range(sizehalf)
        ]
    return [start]*(size-sizehalf)+arr
def SigmoidStrengthModel(start, finish, size):
    def fade_out(x, x1, x2):
        return 1 / (1 + np.exp(-(x - (x1 + x2) / 2) * 8 / (x2 - x1)))
    arr = [start + (finish - start) * (fade_out(i, 0, size) - 0.5) for i in range(size)]
    return arr

class ControlNetContainer:
    def __init__(
            self, controlnet, controlnet_cond, 
            controlnet_gs, controlnet_start_step,
            controlnet_end_step,
            
            ):
        self.controlnet_cond = controlnet_cond
        self.controlnet_gs = controlnet_gs
        self.controlnet_start_step = controlnet_start_step
        self.controlnet_end_step = controlnet_end_step
        self.controlnet = controlnet