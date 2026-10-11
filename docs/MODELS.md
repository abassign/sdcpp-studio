# Models

Model weights are **not** shipped. They are declared in [`../models.yaml`](../models.yaml)
and downloaded on demand into `MODELS_DIR` (a mounted volume).

## Presets

| Preset | What it downloads |
|---|---|
| `demo-min` | SD1.5 realism + a small VLM (I2T/critic/translator) |
| `default` | Z-Image-Turbo + Qwen3-VL-8B |
| `flux` | FLUX.1-dev + VAE + clip_l + t5xxl |
| `ideogram4` | Ideogram 4 (+uncond) + FLUX.2 VAE + Qwen3-VL |
| `qwen-edit` | Qwen-Image-Edit |
| `uncensored` | Uncensored critic/translator LLM |

```bash
./go.sh setup            # demo-min
./go.sh download flux    # any preset id
```

> Fill the `TODO` entries (repos/files/sizes/licenses) in `models.yaml` before release.
