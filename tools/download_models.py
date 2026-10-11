#!/usr/bin/env python3
"""sdcpp-studio - model installer.

Reads models.yaml and downloads what a preset needs into MODELS_DIR.
Weights are never shipped with the app: this is the only way to get them.

    python3 tools/download_models.py --list
    python3 tools/download_models.py --preset demo-min
    python3 tools/download_models.py --component flux1-dev
    python3 tools/download_models.py --preset flux --yes

Environment:
    MODELS_DIR   target root (default: /app/models)
    HF_TOKEN     needed for gated repos (also read from secrets.json)
"""
import argparse
import fnmatch
import json
import os
import shutil
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML missing: run this inside the sdcpp-studio image, or `pip install pyyaml`.")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGISTRY = os.path.join(ROOT, "models.yaml")

TARGETS = {"diffusion": "checkpoints", "vae": "vae",
           "text_encoder": "text_encoders", "llm": "i2t", "lora": "loras"}


def load_registry():
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def models_dir():
    return os.environ.get("MODELS_DIR", "/app/models")


def hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for p in (os.path.join(ROOT, "secrets.json"), "/app/secrets.json"):
        if os.path.exists(p):
            try:
                return json.load(open(p)).get("hf_token")
            except Exception:
                pass
    return None


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def spec_paths(cid, spec):
    """Return (relative_dir, repo, patterns, size_gb, license, gated)."""
    kind = spec.get("kind", "diffusion")
    base = TARGETS.get(kind, "checkpoints")
    sub = spec.get("subdir")
    if sub is None and kind == "llm":
        sub = cid  # LLM/VLM entries need their OWN folder (critic discovery)
    rel = os.path.join(base, sub) if sub else base
    pats = spec.get("file", "*")
    if isinstance(pats, str):
        pats = [pats]
    return rel, spec.get("repo"), pats, spec.get("size_gb", 0), spec.get("license", "unknown"), spec.get("gated", False)


def already_present(target_dir, patterns):
    if not os.path.isdir(target_dir):
        return False
    files = os.listdir(target_dir)
    return all(any(fnmatch.fnmatch(f, p) for f in files) for p in patterns)


def download_one(cid, spec, yes, dry=False):
    rel, repo, pats, size, lic, gated = spec_paths(cid, spec)
    if repo in (None, "", "TODO"):
        print(f"  ! {cid}: registry entry not filled yet (TODO) -> skipped")
        return False
    tdir = os.path.join(models_dir(), rel)
    os.makedirs(tdir, exist_ok=True)
    if already_present(tdir, pats):
        print(f"  = {cid}: already present in {rel}/")
        return True
    print(f"  + {cid}: {repo} {pats}  -> {rel}/  ~{size}GB  license={lic}" + ("  (GATED)" if gated else ""))
    if dry:
        return True
    if gated and not hf_token():
        print("    ! gated repo but no HF_TOKEN set (export HF_TOKEN=... or secrets.json)")
    if not yes:
        try:
            if input("    download? [Y/n] ").strip().lower() not in ("", "y", "yes"):
                print("    skipped")
                return False
        except EOFError:
            pass
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo,
        allow_patterns=pats,
        local_dir=tdir,
        token=hf_token(),
    )
    return True


def main():
    ap = argparse.ArgumentParser(description="sdcpp-studio model installer")
    ap.add_argument("--preset", help="preset id from models.yaml")
    ap.add_argument("--component", help="single component id")
    ap.add_argument("--list", action="store_true", help="list presets/components")
    ap.add_argument("--yes", "-y", action="store_true", help="do not prompt")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    args = ap.parse_args()

    reg = load_registry()
    comps = reg.get("components", {})
    presets = reg.get("presets", {})

    if args.list:
        print("Presets:")
        for pid, p in presets.items():
            print(f"  {pid:12s} {p.get('description','')}")
            print(f"               -> {', '.join(p.get('components', []))}")
        return

    if args.component:
        ids = [args.component]
    elif args.preset:
        if args.preset not in presets:
            sys.exit(f"unknown preset '{args.preset}' (see --list)")
        ids = presets[args.preset].get("components", [])
    else:
        sys.exit("specify --preset <id> or --component <id> (see --list)")

    total = sum(spec_paths(c, comps[c])[3] or 0 for c in ids if c in comps)
    free = shutil.disk_usage(models_dir() if os.path.isdir(models_dir()) else ROOT).free
    print(f"Target: {models_dir()}  |  preset downloads ~{total}GB (free: {human(free)})")
    if total and free < total * 1024**3:
        sys.exit("Not enough disk space for this preset.")

    ok = 0
    for cid in ids:
        if cid not in comps:
            print(f"  ? {cid}: not in components -> skipped")
            continue
        try:
            if download_one(cid, comps[cid], args.yes, args.dry_run):
                ok += 1
        except Exception as e:
            print(f"  ! {cid}: download failed: {e}")
    print(f"Done: {ok}/{len(ids)} components ready.")


if __name__ == "__main__":
    main()
