# Architecture

> TODO

- `api/` - FastAPI routers (generate, models, history, agent, ...)
- `core/` - shared logic (config, model config, text encoders, ...)
- `workers/` - generation workers: `t2i_cpp`, `i2i_cpp`, `i2t_llama`, `agent_refine`, ...
- `static/` - web UI (vanilla JS)
- `tools/` - audit, model downloader, sync
- engines: `sdcpp_src/` (stable-diffusion.cpp) and `llama.cpp/` as git submodules
