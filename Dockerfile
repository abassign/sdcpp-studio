# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# sdcpp-studio - multi-variant image (ROCm / CUDA)
#
#   docker build --build-arg GPU_FLAVOR=rocm -t sdcpp-studio:rocm .
#   docker build --build-arg GPU_FLAVOR=cuda -t sdcpp-studio:cuda \
#       --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04 .
#
# The engines (stable-diffusion.cpp + llama.cpp) are built from the git
# submodules. Model weights are NEVER baked in: they are mounted (MODELS_DIR).
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1

# ===========================================================================
# base: python + system deps
# ===========================================================================
FROM ${BASE_IMAGE} AS base

ARG GPU_FLAVOR=rocm
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ROCM_HOME=/opt/rocm \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib:/usr/local/cuda/lib64

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates patch build-essential python3-dev python3-pip \
        cmake ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# extra ROCm libs (libmtmd etc.) - only for the rocm flavour
RUN if [ "$GPU_FLAVOR" = "rocm" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            rocm-libs hip-runtime-amd miopen-hip \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "numpy<2.0" \
        accelerate sentencepiece tiktoken protobuf timm einops \
        opencv-python-headless pillow regex requests tokenizers \
        filelock safetensors pyyaml huggingface_hub

# torch: comes with the ROCm base image; install it explicitly on CUDA
RUN if [ "$GPU_FLAVOR" = "cuda" ]; then \
        python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124; \
    fi

# ===========================================================================
# engines: build stable-diffusion.cpp (+Krea2 patch) and llama.cpp
# ===========================================================================
FROM base AS engines

ARG GPU_FLAVOR=rocm
ARG AMDGPU_TARGETS=gfx1100

COPY patches/ /app/patches/
COPY sdcpp_src/ /app/sdcpp_src/
RUN cd /app/sdcpp_src && \
    patch -p1 --forward < /app/patches/krea2-lokr.patch && \
    mkdir -p build && cd build && \
    if [ "$GPU_FLAVOR" = "cuda" ]; then \
        cmake .. -DSD_CUBLAS=ON -DSD_FLASH_ATTN=ON -DSD_FAST_SOFTMAX=ON \
                 -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON; \
    else \
        cmake .. -DSD_HIPBLAS=ON -DSD_FLASH_ATTN=ON -DSD_FAST_SOFTMAX=ON \
                 -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=${AMDGPU_TARGETS} \
                 -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON; \
    fi && \
    cmake --build . -j"$(nproc)" && \
    cp bin/sd-cli /usr/local/bin/sd && chmod +x /usr/local/bin/sd

COPY llama.cpp/ /app/llama.cpp/
RUN cd /app/llama.cpp && mkdir -p build && cd build && \
    if [ "$GPU_FLAVOR" = "cuda" ]; then \
        cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release; \
    else \
        cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=${AMDGPU_TARGETS} \
                 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release; \
    fi && \
    cmake --build . -j"$(nproc)" --target llama-cli llama-server llama-mtmd-cli && \
    cp bin/llama-cli bin/llama-server bin/llama-mtmd-cli /usr/local/bin/ && \
    find . -name "libggml*.so*" -exec cp {} /usr/local/lib/ \;

# ===========================================================================
# app: runtime image (no sources, no weights)
# ===========================================================================
FROM base AS app

COPY api/ /app/api/
COPY core/ /app/core/
COPY workers/ /app/workers/
COPY static/ /app/static/
COPY libs/ /app/libs/
COPY tools/ /app/tools/
COPY models.yaml settings.json response_format.gbnf shared_utils.py server.py /app/

COPY --from=engines /usr/local/bin/sd /usr/local/bin/sd
COPY --from=engines /usr/local/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=engines /usr/local/bin/llama-server /usr/local/bin/llama-server
COPY --from=engines /usr/local/bin/llama-mtmd-cli /usr/local/bin/llama-mtmd-cli
COPY --from=engines /usr/local/lib/ /usr/local/lib/

EXPOSE 8000
CMD ["python3", "server.py"]
