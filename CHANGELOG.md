# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-11

### Added

- Local image-generation studio for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  with a FastAPI backend and a vanilla-JS web UI.
- Text-to-Image, Image-to-Image and Image Analysis (I2T), with multiple model
  families (FLUX.1, FLUX.2, Z-Image, Ideogram 4, Qwen-Image, SD/SDXL, ...).
- Inpainting with a built-in **mask editor** (brush / eraser / lasso, dilation +
  feather), consumed by `sd.cpp --mask` on inpaint-capable models.
- **Agentic Refine** loop (goal -> critic -> rewrite -> repeat) with an explore
  mode and a live experience panel.
- Local, **uncensored IT->EN** prompt translation (per-field, selectable model).
- LoRA support and multi-image slots (init / ref / face / mask / control / blend).
- Docker image with **ROCm and CUDA** variants, a `go.sh` launcher and a
  `models.yaml` registry + installer (model weights are never bundled).
- English UI and documentation.

### License

- Released under the **PolyForm Noncommercial License 1.0.0** (noncommercial use
  only; commercial use is not permitted).

[Unreleased]: https://github.com/abassign/sdcpp-studio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/abassign/sdcpp-studio/releases/tag/v0.1.0
