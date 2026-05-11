# ComfyUI-Anima-LLLite

ComfyUI custom node for **ControlNet-LLLite for Anima** (DiT-based).

LLLite is a lightweight ControlNet variant that injects a low-rank correction
into the attention (and optionally MLP) projections of the DiT. This node
loads weights trained with
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and applies them
to the ComfyUI Anima model at inference time.

This is intended as a **minimal reference implementation**. Community nodes
with extra features (per-step scheduling, multi-cond, region masks, per-block
selection, …) are welcome and can build on this codebase.

## Install

Clone into `ComfyUI/custom_nodes/`:

```
cd ComfyUI/custom_nodes
git clone <this-repo> ComfyUI-Anima-LLLite
```

Place LLLite weights (`.safetensors`) under `ComfyUI/models/controlnet/`.

## Node

**Apply Anima ControlNet-LLLite** (`loaders` category)

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | Anima checkpoint |
| `lllite_name` | filename | from `models/controlnet/` |
| `image` | IMAGE | control image (any resolution; auto-resized to latent×8) |
| `strength` | FLOAT | LLLite multiplier (default 1.0) |
| `start_percent` | FLOAT | start of the active sampling window (0.0 = sigma_max) |
| `end_percent` | FLOAT | end of the active sampling window (1.0 = sigma_min) |
| `mask` *(optional)* | MASK | required when the weights are 4-channel (inpaint); white = inpaint area, black = keep |

Output: patched `MODEL`.

All architectural parameters (`cond_emb_dim`, `mlp_dim`, `cond_dim`,
`cond_resblocks`, `use_aspp`, `aspp_dilations`, `target_layers`,
`cond_in_channels`, `inpaint_masked_input`) are baked into the trained
weights and read automatically from the safetensors metadata — they are
not exposed as node inputs because changing them would just cause load
errors.

### Inpainting (4-channel) weights

If the loaded weights were trained with `--lllite_cond_in_channels=4`
(inpaint mode), the node also requires a `MASK` input. The mask convention
matches the sd-scripts training/inference code:

* `white` (1.0) = inpaint area (the region the model should fill)
* `black` (0.0) = keep

The mask is resized to match the control image (nearest-neighbor),
binarized at 0.5, and concatenated with the RGB control image as a 4th
channel (rescaled to `[-1, +1]` to match the RGB range). When the weights
were trained with `--lllite_inpaint_masked_input`, the RGB channels are
also zeroed inside the mask region before concatenation — this is read
from the weights metadata and applied automatically.

3-channel weights continue to work with no mask input; if a mask is
connected to a 3-channel LLLite the node logs a warning and ignores it.
A 4-channel weight without a mask raises a `ValueError` rather than
silently substituting an empty mask.

## How it works

* Discovers the LLLite target Linears under each Anima DiT block according to
  the `target_atomics` recorded in the weight metadata:
  * `self_attn_q_pre` → self-attention `q_proj`
  * `self_attn_kv_pre` → self-attention `k_proj` + `v_proj`
  * `cross_attn_q_pre` → cross-attention `q_proj`
  * `mlp_fc1_pre` → `GPT2FeedForward.layer1` (the MLP fc1)

  The LLM-Adapter sub-tree and `output_proj` are always skipped.
* On each sampling step the wrapper monkey-patches those Linears with the
  LLLite forward, runs `apply_model`, and restores the originals — so the
  patch never leaks across model clones.
* The control image is resized to `latent_HW * 8` once per resolution, then
  embedded by the v2 `conditioning1` trunk (Conv 4×4 s=4 → Conv 3×3 s=1 →
  Conv 4×4 s=4 with GroupNorm/SiLU, optional ResBlocks, optional ASPP, 1×1
  projection, LayerNorm) to a per-token feature map matching the DiT token
  grid. For 4-channel (inpaint) weights, the mask is resized with
  nearest-neighbor, binarized, rescaled to `[-1, +1]`, and concatenated as
  the 4th channel of the cond image before this trunk.
* Each LLLite module applies the v2 forward: `down → silu → concat with
  (cond + per-module depth-embedding) → mid → FiLM(γ, β) → silu → up`,
  added to the original Linear input.
* CFG is supported: `cond_emb` is broadcast to match the runtime batch size.

## Weight format (v2, named keys)

The state dict is in the v2 self-describing format. Each LLLite module's
weights are prefixed by the target Linear's full path, so the file alone
identifies which Linear each tensor belongs to:

```
lllite_conditioning1.{conv1,conv2,conv3,norm1,...,proj,out_norm,...}.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.down.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.mid.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.cond_to_film.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.up.{weight,bias}
lllite_dit_blocks_{i}_self_attn_q_proj.depth_embed
...
```

The expected safetensors metadata keys are:

* `lllite.version` (= `"2"`)
* `lllite.cond_emb_dim`, `lllite.mlp_dim`
* `lllite.cond_dim`, `lllite.cond_resblocks`
* `lllite.use_aspp`, `lllite.aspp_dilations`
* `lllite.target_atomics` (canonical, comma-separated atomic specifiers) —
  falls back to `lllite.target_layers` for older v2 snapshots
* `lllite.cond_in_channels` (`"3"` standard / `"4"` inpaint; defaults to `3`)
* `lllite.inpaint_masked_input` (`"true"` / `"false"`; defaults to `false`,
  only meaningful when `cond_in_channels=4`)

**Legacy weights with `lllite_modules.{i}.*` keys (the pre-v2 format) are
rejected on load.** Re-train against the current sd-scripts codebase.

## Credits

* LLLite design and Anima training implementation: kohya-ss
* Adapted for ComfyUI from `sd-scripts`
* ComfyUI port written collaboratively with Claude (Anthropic, `claude-opus-4-7`)
