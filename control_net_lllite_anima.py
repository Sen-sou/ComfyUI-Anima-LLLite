"""ControlNet-LLLite for Anima (DiT) — ComfyUI port.

Adapted from kohya-ss/sd-scripts. The state_dict layout is identical to the
sd-scripts side, so weights trained with sd-scripts load directly.

Differences vs. the sd-scripts file:
  * No dependency on `library.utils` — uses stdlib logging.
  * Module discovery skips the LLM-Adapter sub-tree by class identity in
    addition to the path-based check (ComfyUI ships two distinct ``Attention``
    classes that share the bare class name).
  * Wrapper is omitted; ComfyUI integrates via ``model_function_wrapper``
    in nodes.py instead.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Class name of the Attention used inside Anima's transformer blocks
# (comfy.ldm.cosmos.predict2.Attention). The LLM-Adapter uses a different
# class with the same bare name; we filter it by path.
TARGET_ATTENTION_CLASS = "Attention"
LLM_ADAPTER_NAME = "llm_adapter"


class LLLiteModuleDiT(nn.Module):
    """Inject the LLLite correction ``x + cx`` into a single Attention Linear."""

    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        # Wrap in a list so the original Linear is not registered as a submodule
        # and its weights stay out of state_dict.
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features

        self.down = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.cond_emb: Optional[torch.Tensor] = None
        self.org_forward = None

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        cx = self.cond_emb  # (B_c, S, cond_emb_dim)

        # Broadcast cond_emb to the runtime batch (CFG cond+uncond, multi-cond).
        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                # Mismatch we can't safely broadcast — fall back to identity.
                return self.org_forward(x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            # Sequence length mismatch — skip rather than crash sampling.
            return self.org_forward(x)

        # Run the LLLite mini-MLP in its own parameter dtype, then cast the
        # correction back to ``x``'s dtype before adding. This is robust to
        # autocast / mixed-precision flows that hand us an ``x`` in a dtype
        # different from the LLLite weights (e.g. bf16 input vs. fp32 LLLite).
        param_dtype = self.down[0].weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)

        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        cx = torch.cat([cx, self.down(x_proc)], dim=-1)
        cx = self.mid(cx)
        if self.dropout is not None and self.training:
            cx = F.dropout(cx, p=self.dropout)
        cx = self.up(cx) * self.multiplier
        if cx.dtype != x.dtype:
            cx = cx.to(x.dtype)
        return self.org_forward(x + cx)


class ControlNetLLLiteDiT(nn.Module):
    """Discovers Anima attention Linears, attaches LLLite modules, manages cond_emb."""

    TARGET_LAYERS_CHOICES = ("self_attn_q", "self_attn_qkv", "self_attn_qkv_cross_q")

    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if target_layers not in self.TARGET_LAYERS_CHOICES:
            raise ValueError(
                f"Unknown target_layers: {target_layers}. choices={self.TARGET_LAYERS_CHOICES}"
            )

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.dropout = dropout
        self.multiplier = multiplier

        # cond image (B,3,H*16,W*16) -> (B, cond_emb_dim, H, W) (stride 16).
        # H,W here are the patchified DiT token grid (= VAE-latent H,W / 2 with patch_spatial=2).
        self.conditioning1 = nn.Sequential(
            nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0),
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, target_layers, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)
        logger.info(
            "ControlNet-LLLite (Anima): created %d modules for target=%s",
            len(self.lllite_modules), target_layers,
        )

    @staticmethod
    def _should_apply(is_self_attn: bool, child_name: str, target_layers: str) -> bool:
        if "output_proj" in child_name:
            return False
        if (not is_self_attn) and (child_name in ("k_proj", "v_proj")):
            return False  # cross_attn K,V live in the text embedding space

        if target_layers == "self_attn_q":
            return is_self_attn and child_name == "q_proj"
        if target_layers == "self_attn_qkv":
            return is_self_attn and child_name in ("q_proj", "k_proj", "v_proj")
        if target_layers == "self_attn_qkv_cross_q":
            if is_self_attn and child_name in ("q_proj", "k_proj", "v_proj"):
                return True
            if (not is_self_attn) and child_name == "q_proj":
                return True
            return False
        raise ValueError(f"Unknown target_layers: {target_layers}")

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        target_layers: str,
        dropout: Optional[float],
        multiplier: float,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        for name, module in dit.named_modules():
            if module.__class__.__name__ != TARGET_ATTENTION_CLASS:
                continue
            if LLM_ADAPTER_NAME in name:
                continue
            # The Anima-block Attention exposes is_selfattn; the LLM-Adapter
            # Attention does not — skip the latter even if path filter misses.
            if not hasattr(module, "is_selfattn"):
                continue
            is_self_attn = bool(module.is_selfattn)

            for child_name, child in module.named_children():
                if not isinstance(child, nn.Linear):
                    continue
                if not self._should_apply(is_self_attn, child_name, target_layers):
                    continue
                full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                )
        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        """cond_image: (B, 3, H*16, W*16) in [-1, 1]; ``None`` clears."""
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)  # (B, C, H, W)
        b, c, h, w = cx.shape
        cx = cx.view(b, c, h * w).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")
    info = lllite.load_state_dict(weights_sd, strict=strict)
    logger.info("loaded LLLite weights from %s: %s", file, info)
    return info


def read_lllite_metadata(file: str) -> dict:
    if os.path.splitext(file)[1] != ".safetensors":
        return {}
    from safetensors import safe_open
    with safe_open(file, framework="pt") as f:
        meta = f.metadata()
    return meta or {}
