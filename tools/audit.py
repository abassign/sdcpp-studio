#!/usr/bin/env python3
"""sdcpp-studio - pre-publish AUDIT.

Scansiona i file tracciati da git e produce un report Markdown su cosa va
sistemato prima di pubblicare:
  1. inventario per estensione
  2. testo probabilmente in italiano (euristica)  -> per la traduzione
  3. path assoluti/personali hardcoded
  4. pattern "segreti" (token, chiavi)
  5. file legacy / morte (funzioni, backup, duplicati) - tracked + untracked noti

Uso:  python3 tools/audit.py > docs/AUDIT.md
"""
import os
import re
import subprocess
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXT_EXT = {".py", ".js", ".css", ".html", ".md", ".sh", ".json",
            ".yml", ".yaml", ".txt", ".gbnf", ".env", ".rules", ".nvidia"}

ITALIAN = re.compile(
    r"\b(della|delle|degli|dello|dalla|dalle|nella|negli|che|non|con|per|una|uno|"
    r"sono|questo|questa|questi|queste|immagine|immagini|errore|finestra|cartella|"
    r"pulsante|senza|anche|solo|quando|perch[ée]|gi[àa]|pi[uù]|pu[oò]|deve|devi|viene|"
    r"stato|fare|tra|fra|molto|dal|del|dei|gli|ancora|adesso|prima|dopo|tutti|tutte)\b",
    re.IGNORECASE)
ACCENT = re.compile(r"[àèéìòùÀÈÉÌÒÙ]")
PATHY = re.compile(r"(/run/media/[^\s\"'`]+|/home/[^\s\"'`]+|/workspace[^\s\"'`]*|"
                   r"100\.66\.107\.106|pclab01|/mnt/Salvataggi[^\s\"'`]*)")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{10,}|hf_token|civitai_token|api[_-]?key|"
                    r"BEGIN [A-Z ]*PRIVATE KEY)", re.IGNORECASE)

LEGACY_DIRS = [".backup_2026-08-31", ".cleanup_backup"]


def tracked_files():
    out = subprocess.check_output(["git", "-C", ROOT, "ls-files"], text=True)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def read(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def main():
    files = tracked_files()
    ext = Counter(os.path.splitext(f)[1].lower() or "(none)" for f in files)

    print("# sdcpp-studio - Pre-publish audit\n")
    print(f"Tracked files: **{len(files)}**  ")
    print(f"Repo root: `{ROOT}`\n")

    print("## 1. Inventory by extension\n")
    print("| Ext | Files |")
    print("|---|---|")
    for e, n in ext.most_common():
        print(f"| `{e or '(none)'}` | {n} |")
    print()

    ital = {}
    ital_tot = 0
    paths = defaultdict(set)
    secrets = []
    for f in files:
        if os.path.splitext(f)[1].lower() not in TEXT_EXT:
            continue
        txt = read(os.path.join(ROOT, f))
        if not txt:
            continue
        c = len(ITALIAN.findall(txt)) + len(ACCENT.findall(txt))
        if c:
            ital[f] = c
            ital_tot += c
        for m in PATHY.findall(txt):
            paths[f].add(m)
        for m in SECRET.findall(txt):
            secrets.append((f, m))

    print("## 2. Suspected Italian text (heuristic)\n")
    print(f"Total Italian tokens: **{ital_tot}** across **{len(ital)}** files."
          "  (UI strings + comments; prioritise user-facing files.)\n")
    print("| File | tokens |")
    print("|---|---|")
    for f, c in sorted(ital.items(), key=lambda x: -x[1])[:40]:
        print(f"| `{f}` | {c} |")
    if len(ital) > 40:
        print(f"| ... +{len(ital)-40} more | |")
    print()

    print("## 3. Hardcoded personal / absolute paths\n")
    if paths:
        print("| File | matches |")
        print("|---|---|")
        for f, ms in sorted(paths.items()):
            print(f"| `{f}` | {', '.join('`'+m+'`' for m in sorted(ms))} |")
    else:
        print("(none)")
    print()

    print("## 4. Secret-ish patterns\n")
    if secrets:
        for f, m in secrets:
            print(f"- `{f}`: `{m[:60]}`")
    else:
        print("(none)")
    print()

    print("## 5. Legacy / dead files\n")
    removed = [f for f in files if f.endswith(".REMOVED")]
    print(f"Tracked `*.REMOVED` files: **{len(removed)}**")
    for f in removed:
        print(f"- `{f}`")
    for d in LEGACY_DIRS:
        if os.path.isdir(os.path.join(ROOT, d)):
            print(f"- untracked legacy dir: `{d}/`")
    for extra in ("scaffold.py", "gonvidia.sh", "venv", ".venv"):
        if os.path.exists(os.path.join(ROOT, extra)):
            print(f"- present (review): `{extra}`")
    print()

    print("## 6. Notes\n")
    print("- `secrets.json` is gitignored (verify it is never tracked).")
    print("- Model weights / caches must never be committed (see .gitignore).")
    print("- `sdcpp_src/` and `llama.cpp/` are NOT tracked -> externalise them "
          "(submodule or fetch at build).")


if __name__ == "__main__":
    main()
