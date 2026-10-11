# sdcpp-studio - Pre-publish audit

Tracked files: **151**  
Repo root: `/run/media/abassign/AI2/docker/sdcpp-studio`

## 1. Inventory by extension

| Ext | Files |
|---|---|
| `.py` | 71 |
| `.md` | 23 |
| `.js` | 18 |
| `(none)` | 17 |
| `.css` | 7 |
| `.sh` | 5 |
| `.gbnf` | 2 |
| `.json` | 2 |
| `.html` | 2 |
| `.nvidia` | 1 |
| `.yml` | 1 |
| `.txt` | 1 |
| `.typed` | 1 |

## 2. Suspected Italian text (heuristic)

Total Italian tokens: **1644** across **78** files.  (UI strings + comments; prioritise user-facing files.)

| File | tokens |
|---|---|
| `docs/AGENT_REFINE_EXTRA_FIELDS.md` | 244 |
| `plans/multi-image-integration.md` | 157 |
| `AI-CONFIGURATION.md` | 113 |
| `go.sh` | 111 |
| `plans/agent_refine_review.md` | 96 |
| `plans/multi-image-integration-ui.md` | 93 |
| `README.md` | 90 |
| `plans/semantic_clustering_plan.md` | 70 |
| `autoconfig.py` | 65 |
| `plans/multi-image-n-slots.md` | 64 |
| `scaffold.py` | 56 |
| `static/js/upload.js` | 51 |
| `plans/i2t-validation-plan.md` | 36 |
| `logs/project_log.md` | 35 |
| `plans/model-grouping-plan.md` | 33 |
| `static/js/models.js` | 33 |
| `static/index.html` | 32 |
| `static/js/mask_editor.js` | 27 |
| `workers/agent_refine.py` | 20 |
| `static/js/agent.js` | 16 |
| `static/js/translate.js` | 14 |
| `api/history.py` | 9 |
| `core/models.py` | 9 |
| `gonvidia.sh` | 9 |
| `test_i2t_docker.sh` | 9 |
| `libs/gguf/scripts/gguf_editor_gui.py` | 8 |
| `static/css/panels.css` | 8 |
| `static/js/history.js` | 8 |
| `api/agent.py` | 7 |
| `static/js/init.js` | 7 |
| `workers/t2i_cpp.py` | 7 |
| `api/generate.py` | 6 |
| `api/llama_server_manager.py` | 6 |
| `static/js/generate.js` | 6 |
| `api/config.py` | 5 |
| `workers/i2t_llama.py` | 5 |
| `libs/gguf/lazy.py` | 4 |
| `shared_utils.py` | 4 |
| `static/js/config.js` | 4 |
| `static/js/lora.js` | 4 |
| ... +38 more | |

## 3. Hardcoded personal / absolute paths

| File | matches |
|---|---|
| `README.md` | `/home/user/my_models`, `/home/user/my_outputs` |
| `api/config.py` | `/home/user/models`, `/home/user/models)`, `/home/user/models).` |
| `autoconfig.py` | `/home/abassign/docker/models`, `/mnt/Salvataggi/Salvataggi/docker/models` |
| `docker-compose.yml` | `/home/abassign/docker/models/checkpoints:/app/models_external`, `/home/abassign/docker/outputs:/app/outputs` |
| `docs/GLM-4.1V-Setup.md` | `/home/user/my_models`, `/home/user/my_outputs` |
| `logs/project_log.md` | `/home/abassign/.pi)`, `/home/abassign/docker/models`, `/run/media/abassign/AI1/docker/models`, `/workspace`, `/workspace/.pi/skills/autoconfigure`, `/workspace/.pi/skills/install-model` |
| `plans/model-grouping-plan.md` | `/home/abassign/docker/models`, `/home/abassign/docker/models/` |
| `static/setup.html` | `/home/user/${folderName}`, `/home/user/${handle.name}`, `/home/user/${selectedPath}` |
| `test_i2t_grammar.sh` | `/home/abassign/docker/models}`, `/home/abassign/docker/outputs}` |
| `workers/i2i_cpp.py` | `/workspace/...):` |
| `workers/t2i_cpp.py` | `/workspace/...):` |

## 4. Secret-ish patterns

- `api/config.py`: `hf_token`
- `api/config.py`: `civitai_token`
- `api/config.py`: `hf_token`
- `api/config.py`: `hf_token`
- `api/config.py`: `civitai_token`
- `api/config.py`: `civitai_token`
- `api/config.py`: `hf_token`
- `api/config.py`: `hf_token`
- `api/config.py`: `hf_token`
- `api/config.py`: `civitai_token`
- `api/config.py`: `civitai_token`
- `api/config.py`: `civitai_token`
- `api/models.py`: `civitai_token`
- `api/models.py`: `hf_token`
- `download_zimage_deps.py`: `HF_TOKEN`
- `libs/gguf/metadata.py`: `hf_parameters`
- `libs/gguf/metadata.py`: `hf_parameters`
- `libs/gguf/utility.py`: `HF_TOKEN`
- `libs/gguf/utility.py`: `HF_TOKEN`
- `libs/gguf/vocab.py`: `hf_token`
- `libs/gguf/vocab.py`: `hf_token`
- `scaffold.py`: `hf_token`
- `scaffold.py`: `hf_token`
- `scaffold.py`: `hf_token`
- `static/setup.html`: `hf_token`
- `static/setup.html`: `civitai_token`
- `static/setup.html`: `hf_token`
- `static/setup.html`: `civitai_token`
- `static/setup.html`: `hf_token`
- `static/setup.html`: `civitai_token`
- `static/setup.html`: `hf_token`
- `static/setup.html`: `civitai_token`
- `static/setup.html`: `hf_token`
- `static/setup.html`: `civitai_token`
- `verify_keys.py`: `hf_token`
- `verify_keys.py`: `civitai_token`
- `verify_keys.py`: `hf_token`
- `verify_keys.py`: `hf_token`
- `workers/i2i_faceswap.py`: `hf_token`

## 5. Legacy / dead files

Tracked `*.REMOVED` files: **0**
- untracked legacy dir: `.backup_2026-08-31/`
- untracked legacy dir: `.cleanup_backup/`
- present (review): `scaffold.py`
- present (review): `gonvidia.sh`
- present (review): `venv`
- present (review): `.venv`

## 6. Notes

- `secrets.json` is gitignored (verify it is never tracked).
- Model weights / caches must never be committed (see .gitignore).
- `sdcpp_src/` and `llama.cpp/` are NOT tracked -> externalise them (submodule or fetch at build).
