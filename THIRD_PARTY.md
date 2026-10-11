# Third-party components and licenses

`sdcpp-studio` itself is licensed under **PolyForm Noncommercial 1.0.0**
(see [LICENSE](LICENSE)). It builds on, links to, or downloads third-party
components that keep **their own licenses**. This file lists the main ones.

> You must comply with each component's license in addition to ours.
> Model weights are **not** distributed by this project.

## Engines (built from git submodules)

| Component | License | Copyright |
|---|---|---|
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | MIT | leejet and contributors |
| [ggml](https://github.com/ggml-org/ggml) (submodule of sd.cpp) | MIT | ggml authors |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | ggml authors |

The file `patches/krea2-lokr.patch` is a **local modification** of
`src/model/adapter/lora_ops.cpp` from stable-diffusion.cpp, therefore it is
distributed under the **same MIT license** as that project.

## Vendored Python package

| Component | License |
|---|---|
| `libs/gguf` (GGUF reader, from `gguf` PyPI package) | MIT |

## Runtime Python dependencies (installed via pip, not vendored)

- FastAPI, uvicorn, pydantic - MIT / BSD
- PyTorch - BSD-3-Clause
- NumPy - BSD-3-Clause
- Pillow - HPND
- huggingface_hub, safetensors, tokenizers - Apache-2.0
- opencv-python-headless - Apache-2.0
- tiktoken, protobuf, timm, einops, sentencepiece - MIT / Apache-2.0 / BSD

Refer to each package for the exact terms.

## Model weights (downloaded separately, NOT part of this repo)

Model weights have their **own licenses** and are **not** covered by our
license. Examples of families this app can use:

| Model family | Typical license |
|---|---|
| FLUX.1 / FLUX.2 | FLUX.1 [dev] Non-Commercial License |
| Ideogram 4 | Ideogram License |
| Z-Image | see model card |
| Qwen-Image / Qwen-VL | Apache-2.0 |
| Stable Diffusion / SDXL | CreativeML Open RAIL-M |
| Gemma | Gemma Terms of Use |

**Always check the license of every model you download and use.** Some are
non-commercial, some require acceptance on Hugging Face (gated), some allow
commercial use. The app does not change or relicense model weights.
