# sdcpp-studio

[![CI](https://github.com/abassign/sdcpp-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/abassign/sdcpp-studio/actions/workflows/ci.yml)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)

A local, private **image-generation studio** built on
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Text-to-image, image-to-image, image analysis (I2T), inpainting with a built-in
mask editor, an agentic refinement loop, local translation and more - all running
on your own machine (AMD ROCm or NVIDIA CUDA), with no cloud calls.

> **Unofficial.** Not affiliated with or endorsed by the
> `stable-diffusion.cpp`, FLUX, Ideogram, Qwen or Z-Image projects.

## Features

- Text-to-Image (T2I) and Image-to-Image (I2I) with many model families
  (FLUX.1, FLUX.2, Z-Image, Ideogram 4, Qwen-Image-Edit, SD/SDXL, ...)
- Image Analysis (I2T) with local VLMs
- **Agentic Refine** loop (goal -> critic -> rewrite -> repeat, with explore mode)
- **Inpainting**: paint/lasso mask editor with dilation + feather (`--mask`)
- LoRA support, mask/face/control/blend slots
- Local (uncensored) IT->EN translation for prompts
- Runs on **AMD ROCm** and **NVIDIA CUDA**

## Quick start

```bash
git clone <this-repo> sdcpp-studio && cd sdcpp-studio
cp .env.example .env          # optional: set MODELS_DIR / tokens
./go.sh                       # prepare + start the web UI
```

`./go.sh` will:
1. build/pull the Docker image,
2. install the **demo-min** model preset on first run (if none are present),
3. start the server at http://localhost:8000.

Useful commands:

```bash
./go.sh                 # start (context-aware data dirs)
./go.sh setup           # download the demo-min model preset
./go.sh download flux   # download another preset (see models.yaml)
./go.sh --stop          # stop the container (frees GPU/VRAM)
./go.sh --stop-all      # stop every GPU container
./go.sh shell           # interactive shell in the container
```

## Models

Model **weights are never shipped**: they live in `MODELS_DIR` (a mounted
volume) and are fetched on demand. See [`models.yaml`](models.yaml) for the
registry and presets, and [`docs/MODELS.md`](docs/MODELS.md).

Each model keeps its own license (FLUX, Ideogram, Z-Image, Qwen, ...); check it
before use/redistribution.

## Requirements

- Docker + a GPU with a working ROCm or CUDA setup (CPU fallback is slow).
- VRAM: ~8 GB for the demo preset, **24 GB** recommended for FLUX/Ideogram.
- Disk: ~10 GB (demo) up to ~60 GB (full set).

## License

Our code is released under the **PolyForm Noncommercial License 1.0.0**
(see [LICENSE](LICENSE)):

- ✅ free to use, study, modify and share for **noncommercial purposes**
  (personal, hobby, research, education, nonprofits, ...)
- ❌ **commercial use is not permitted**

> Note: a noncommercial license is **not** an OSI "open source" license - that
> is intentional. The `stable-diffusion.cpp` and `llama.cpp` engines are **MIT**,
> and each **model weight** keeps its own license (FLUX, Ideogram, Z-Image, Qwen,
> SD/SDXL, ...). See [THIRD_PARTY.md](THIRD_PARTY.md).

## Docs

- [`docs/INSTALL.md`](docs/INSTALL.md) - detailed install / GPU setup
- [`docs/MODELS.md`](docs/MODELS.md) - model registry & presets
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - how it works
- [`docs/PLAN.md`](docs/PLAN.md) - project roadmap
