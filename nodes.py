"""ComfyUI node for Anima ControlNet-LLLite.

Single LoRA-style node: takes a MODEL, an LLLite weights file, a control IMAGE
and a strength; returns the patched MODEL. Integration is done via
``set_model_unet_function_wrapper`` so the LLLite contribution is fully scoped
to this model clone — no global monkey-patching that could leak into other
samplers in the same workflow.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

import folder_paths

from .control_net_lllite_anima import (
    ControlNetLLLiteDiT,
    load_lllite_weights,
    read_lllite_metadata,
)

logger = logging.getLogger(__name__)


def _get_inner_dit(model) -> torch.nn.Module:
    """Reach the underlying Anima DiT (nn.Module) from a ComfyUI ModelPatcher."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Input MODEL has no .model attribute (not a ModelPatcher?)")
    dit = getattr(inner, "diffusion_model", None)
    if dit is None:
        raise RuntimeError("MODEL.model has no .diffusion_model — not a UNet/DiT model?")
    return dit


def _prepare_cond_image(image: torch.Tensor, latent_h: int, latent_w: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,3) in [0,1] → (1,3,H*8,W*8) in [-1,1].

    The LLLite ``conditioning1`` Conv has stride 16, so the cond image must be
    sized to ``latent_HW * 8`` in input pixel space (= ``token_HW * 16`` after
    DiT patchify with patch_spatial=2).
    """
    if image.ndim == 4 and image.shape[-1] == 3:
        # (B, H, W, 3) -> (B, 3, H, W)
        img = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)} (expected B,H,W,3)")

    img = img[:1]  # use first frame only
    target_h = latent_h * 8
    target_w = latent_w * 8
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)


class AnimaLLLiteApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lllite_name": (folder_paths.get_filename_list("controlnet"),),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "cond_emb_dim": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1,
                                          "tooltip": "0 = read from weights metadata (default 32)"}),
                "mlp_dim": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1,
                                     "tooltip": "0 = read from weights metadata (default 64)"}),
                "target_layers": (["auto", "self_attn_q", "self_attn_qkv", "self_attn_qkv_cross_q"],
                                  {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, lllite_name, image, strength,
              cond_emb_dim=0, mlp_dim=0, target_layers="auto"):
        weights_path = folder_paths.get_full_path("controlnet", lllite_name)
        if weights_path is None or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LLLite weights not found: {lllite_name}")

        meta = read_lllite_metadata(weights_path)
        ce_dim = cond_emb_dim if cond_emb_dim > 0 else int(meta.get("lllite.cond_emb_dim", 32))
        m_dim = mlp_dim if mlp_dim > 0 else int(meta.get("lllite.mlp_dim", 64))
        tl = target_layers if target_layers != "auto" else meta.get("lllite.target_layers", "self_attn_q")

        dit = _get_inner_dit(model)
        lllite = ControlNetLLLiteDiT(
            dit,
            cond_emb_dim=ce_dim,
            mlp_dim=m_dim,
            target_layers=tl,
            multiplier=strength,
        )
        load_lllite_weights(lllite, weights_path, strict=False)
        lllite.eval().requires_grad_(False)

        # Capture image tensor (cloned to detach from any upstream caching)
        src_image = image.detach().clone()

        # Cache for the per-resolution preprocessed cond image (avoids repeat resize)
        cache = {"cond_image_pp": None, "key": None, "lllite_loaded_to": None}

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Anima latent shape: (B, C, T, H, W) — take spatial dims from the tail.
            latent_h, latent_w = int(input_x.shape[-2]), int(input_x.shape[-1])
            device = input_x.device
            dtype = input_x.dtype

            # Move LLLite to the runtime device/dtype lazily.
            tag = (device, dtype)
            if cache["lllite_loaded_to"] != tag:
                lllite.to(device=device, dtype=dtype)
                cache["lllite_loaded_to"] = tag
                cache["cond_image_pp"] = None  # invalidate

            key = (latent_h, latent_w, device, dtype)
            if cache["key"] != key or cache["cond_image_pp"] is None:
                cache["cond_image_pp"] = _prepare_cond_image(
                    src_image, latent_h, latent_w, device, dtype
                )
                cache["key"] = key

            lllite.set_multiplier(strength)
            lllite.set_cond_image(cache["cond_image_pp"])
            lllite.apply_to()
            try:
                return apply_model(input_x, timestep, **c)
            finally:
                lllite.restore()
                lllite.clear_cond_image()

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaLLLiteApply": AnimaLLLiteApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLLLiteApply": "Apply Anima ControlNet-LLLite",
}
