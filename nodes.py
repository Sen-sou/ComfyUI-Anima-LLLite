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
    ASPP_DEFAULT_DILATIONS,
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
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, lllite_name, image, strength, start_percent, end_percent):
        weights_path = folder_paths.get_full_path("controlnet", lllite_name)
        if weights_path is None or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LLLite weights not found: {lllite_name}")

        # Architecture is fully determined by the trained weights — read everything
        # from metadata rather than exposing knobs that would just cause load errors.
        meta = read_lllite_metadata(weights_path)
        ce_dim = int(meta.get("lllite.cond_emb_dim", 32))
        m_dim = int(meta.get("lllite.mlp_dim", 64))
        # v2 records the canonical atomic form under lllite.target_atomics; fall back
        # to the legacy preset key, then to the v1 default.
        tl = meta.get("lllite.target_atomics", meta.get("lllite.target_layers", "self_attn_q"))
        cond_dim = int(meta.get("lllite.cond_dim", 64))
        cond_resblocks = int(meta.get("lllite.cond_resblocks", 1))
        use_aspp = str(meta.get("lllite.use_aspp", "false")).lower() == "true"
        aspp_dilations_meta = meta.get("lllite.aspp_dilations")
        if use_aspp and aspp_dilations_meta:
            aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
        else:
            aspp_dilations = ASPP_DEFAULT_DILATIONS

        dit = _get_inner_dit(model)
        lllite = ControlNetLLLiteDiT(
            dit,
            cond_emb_dim=ce_dim,
            mlp_dim=m_dim,
            target_layers=tl,
            multiplier=strength,
            cond_dim=cond_dim,
            cond_resblocks=cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
        )
        load_lllite_weights(lllite, weights_path, strict=False)
        lllite.eval().requires_grad_(False)

        # Convert percent range -> sigma range (start_percent=0 → sigma_max).
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        # Capture image tensor (cloned to detach from any upstream caching)
        src_image = image.detach().clone()

        # Cache for the per-resolution preprocessed cond image (avoids repeat resize)
        cache = {"cond_image_pp": None, "key": None, "lllite_loaded_to": None}

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Step-range gate: skip LLLite entirely when current sigma is outside
            # [sigma_end, sigma_start]. percent_to_sigma maps 0.0 → sigma_max,
            # 1.0 → sigma_min, so the active window is sigma_end <= sigma <= sigma_start.
            sigma = float(timestep.max().item())
            if not (sigma_end <= sigma <= sigma_start):
                return apply_model(input_x, timestep, **c)

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
