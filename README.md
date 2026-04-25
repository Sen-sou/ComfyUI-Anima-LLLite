# ComfyUI-Anima-LLLite

ComfyUI custom node for **ControlNet-LLLite for Anima** (DiT-based).

LLLite is a lightweight ControlNet variant that injects a low-rank correction
into the attention projections of the DiT. This node loads weights trained with
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and applies them
to the ComfyUI Anima model at inference time.

This is intended as a **minimal reference implementation**. Community nodes
with extra features (per-step scheduling, multi-cond, region masks, …) are
welcome and can build on this codebase.

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
| `cond_emb_dim` / `mlp_dim` / `target_layers` | optional | overrides for weights without metadata |

Output: patched `MODEL`.

## How it works

* Discovers `q_proj` / `k_proj` / `v_proj` Linears under each Attention block of
  the Anima DiT (skipping the LLM-Adapter and `output_proj`).
* On each sampling step the wrapper monkey-patches those Linears with the LLLite
  forward, runs `apply_model`, and restores the originals — so the patch never
  leaks across model clones.
* The control image is resized to `latent_HW * 8` once per resolution, then
  embedded by `conditioning1` (stride-16 Conv stack) to a per-token feature map
  matching the DiT token grid.
* CFG is supported: `cond_emb` is broadcast to match the runtime batch size.

## Weight format

State dict keys are the same as sd-scripts:
```
conditioning1.{0,2}.{weight,bias}
lllite_modules.{i}.down.0.{weight,bias}
lllite_modules.{i}.mid.0.{weight,bias}
lllite_modules.{i}.up.{weight,bias}
```

If the safetensors file has metadata (`lllite.cond_emb_dim`, `lllite.mlp_dim`,
`lllite.target_layers`), the node reads it automatically; otherwise specify the
values via the optional inputs.

## Credits

* LLLite design and Anima training implementation: kohya-ss
* Adapted for ComfyUI from `sd-scripts`
* ComfyUI port written collaboratively with Claude (Anthropic, `claude-opus-4-7`)
