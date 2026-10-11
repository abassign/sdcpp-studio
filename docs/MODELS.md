# Models

Model weights are **not** shipped with the app. They are declared in
[`../models.yaml`](../models.yaml) and downloaded on demand into `MODELS_DIR`
(a mounted volume).

```bash
./go.sh list                    # show presets
./go.sh setup                   # download the demo-min preset
./go.sh download <preset>       # download another preset
```

`./go.sh setup` runs the installer **inside the Docker image** (it has
`huggingface_hub`), so no host Python setup is needed.

## Minimal install (`demo-min`) — makes the app runnable in ~5 GB

| Component | Purpose | Repo | Repo size | Installed to |
|---|---|---|---|---|
| **SD 1.5** (`v1-5-pruned-emaonly-fp16`) | T2I / I2I (self-contained: VAE + text encoders embedded) | `Comfy-Org/stable-diffusion-v1-5-archive` | ~2.0 GB | `checkpoints/sd15/` |
| **Qwen3-VL-4B abliterated** + `mmproj` | I2T / agent critic / translator (vision) | `mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF` | ~3.2 GB | `i2t/Qwen3-VL-4B-abliterated/` |

Total **~5.2 GB**. With just these two, you get generation (T2I/I2I), image
analysis (I2T), the agentic refine loop and local translation. T2I/I2I work with
**no** extra VAE/text-encoder files because the SD1.5 checkpoint is
self-contained.

## All presets

| Preset | What it downloads | Notes |
|---|---|---|
| `demo-min` | SD 1.5 + Qwen3-VL-4B (+mmproj) | ~5 GB, runnable |
| `default` | Z-Image-Turbo + its VAE + text encoder + Qwen3-VL-4B | needs a ~24 GB GPU (some entries still `TODO`) |
| `flux` | FLUX.1-dev + VAE + clip_l + t5xxl + Qwen3-VL-4B | inpaint (`--mask`); FLUX.1-dev is gated (accept the license, set `HF_TOKEN`) |
| `ideogram4` | Ideogram 4 (+uncond) + FLUX.2 VAE + Qwen3-VL encoder + Qwen3-VL-4B | best text rendering |
| `qwen21` | Qwen-Image 2.1 + VAE + Qwen3-VL-4B | — |

## Layout expected by the app

```
MODELS_DIR/
  checkpoints/<name>/   *.gguf, *.safetensors   (T2I/I2I models)
  vae/                  *vae*.safetensors
  text_encoders/        clip_l, t5xxl, qwen*vl*
  i2t/<model>/          one FOLDER per VLM: <model>.gguf + <model>.mmproj.gguf
```

> The I2T/critic discovery scans **one folder per VLM** under `i2t/`, and pairs
> the `.gguf` with the matching `.mmproj*.gguf` inside that folder.

## Notes

- Model weights keep **their own licenses** (see [THIRD_PARTY.md](../THIRD_PARTY.md)).
- Gated repos (e.g. FLUX.1-dev) need `HF_TOKEN` (put it in `.env` or `secrets.json`).
- Entries marked `TODO` in `models.yaml` have not been verified yet.
