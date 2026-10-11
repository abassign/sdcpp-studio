# sdcpp-studio - pre-publish plan

> **Name:** `sdcpp-studio` - a local studio for stable-diffusion.cpp.
> *Unofficial; not affiliated with leejet/stable-diffusion.cpp.*
>
> Goal: turn the current working app into a clean, English, publishable project
> where models are installed (not shipped) and Docker handles setup.

Status: **PLANNING** - no code moved yet. Grounded on `docs/AUDIT.md`.

---

## 0. Decisions (CONFIRMED)

| # | Decision | Chosen |
|---|---|---|
| 1 | GPU targets | **ROCm + NVIDIA/CUDA** (as already set up); CPU best-effort |
| 2 | Language | **English only** (no i18n for 1.0) |
| 3 | Publish target | GitHub **`abassign/sdcpp-studio`** (new repo) |
| 4 | Repo strategy | **new repo**; `pclab01` stays as the current backup, later it will sync **from the new GitHub repo** |
| 5 | License | MIT for *our* code; model weights keep their own licenses |

## 0b. Target repo layout

```
sdcpp-studio/
  README.md  LICENSE  CONTRIBUTING.md  .env.example  .gitignore
  Dockerfile                 # multi-target: rocm / cuda / cpu
  docker-compose.yml         # volumes only: models/ outputs/ hf_cache/
  go.sh                      # setup / download <preset> / build / shell / --stop
  models.yaml                # model registry + presets
  api/          core/        # FastAPI routers + shared logic
  workers/                   # sd.cpp + llama workers
  static/                    # frontend (index.html, js, css)
  tools/                     # audit.py, model downloader, sync
  docs/                      # PLAN, AUDIT, ARCHITECTURE, MODELS, INSTALL
  tests/
  sdcpp_src/     (submodule) # leejet/stable-diffusion.cpp (pinned)
  llama.cpp/     (submodule) # ggml-org/llama.cpp (pinned)
  libs/                      # vendored gguf helpers (keep) or submodule
  # runtime (gitignored, mounted volumes):
  models/  outputs/  hf_cache/  data/
```

Note: the new repo and the current working dir share the name `sdcpp-studio`;
we keep the same layout so the transition is a copy + cleanup, not a rewrite.

---

## 1. Baseline & skeleton

- Tag/backup the current working tree; keep `pclab01` as the live copy.
- Create a **new repo** `sdcpp-studio` with: `.gitignore`, `LICENSE` (MIT),
  `README`, `CONTRIBUTING`, `docs/`, CI (lint + build smoke).
- **Do not translate in place**: structure first, text second.

## 2. Externalise the C++ engines

Audit: `sdcpp_src/` and `llama.cpp/` have **0 tracked files** today.

- Add as **git submodules** (pinned commits) *or* fetch+patch during the Docker build:
  - `leejet/stable-diffusion.cpp`
  - `ggml-org/llama.cpp`
- Remove from the repo: `sdcpp_src/`, `llama.cpp/`, `llama-*.tar.gz`, the `sd` binary.
- Keep the local build patch for the Krea2 LoKr fix documented (apply as a patch file).

## 3. Docker + model installation (highest priority)

- **Multi-target Dockerfile**: `rocm` / `cuda` / `cpu`, multi-stage.
- **docker-compose.yml** with volumes only: `models/`, `outputs/`, `hf_cache/`.
  - Fix today's hardcoded paths (audit: `docker-compose.yml`, `autoconfig.py`).
- **No weights in the image** (<- non-negotiable: licenses + size).
- **`models.yaml` registry** (already drafted) + installer:
  - `./go.sh setup`      -> download the **demo-min** preset (few GB)
  - `./go.sh download <preset>` -> `default` / `flux` / `ideogram4` / `uncensored`
  - pre-flight: disk space, optional SHA256, license acknowledgement.
- Document VRAM per preset (e.g., 24 GB for FLUX/ID4).

## 4. Config & cleanup

- Replace hardcoded paths: `/app`, `/run/media/abassign`, `/workspace`
  (audit: `autoconfig.py`, `workers/*_cpp.py` comments, `logs/`, `plans/`).
- `.env.example` for tokens (HF/Civitai). `secrets.json` stays gitignored.
- Remove untracked legacy: `.backup_2026-08-31/`, `.cleanup_backup/`,
  old `scaffold.py`, `gonvidia.sh` (superseded by `go.sh` GPU autodetect).

## 5. Italian -> English (batched)

Audit: **1644 Italian tokens across 78 files**. Order:

1. **UI strings** - `static/index.html`, `static/js/*.js`, `static/css/*.css`
2. **Messages / errors** - `api/*.py`, `workers/*.py`, `go.sh`
3. **Comments** - `core/*.py`, `workers/*.py`
4. **Docs** - `README.md`, `docs/`, `plans/` (drop or archive obsolete plans)

Use a **glossary** for recurring terms (go.sh, slot, mask, inpaint, goal,
commit step, ...) to stay consistent. Tooling: `tools/audit.py` to track progress.

## 6. Docs & publish

- `README.md` (EN) with quick start, `INSTALL.md`, `MODELS.md`, `ARCHITECTURE.md`.
- Publish images to **GHCR** (`ghcr.io/<user>/sdcpp-studio:<variant>`).
- Do **not** push to GitHub `abassign/GLM-Image-Studio` (obsolete, frozen).

---

## Recommended order

```
0 decisions  ->  1 skeleton  ->  2 C++ submodules  ->  3 Docker+models
     ->  4 cleanup  ->  5 translation (UI first)  ->  6 docs+publish
```

## Next concrete step

Provide answers to section 0, then:
- fill `models.yaml` TODOs (verify repos/files/sizes/licenses),
- run `tools/audit.py` after each batch to measure translation progress.
