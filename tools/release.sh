#!/usr/bin/env bash
#
# tools/release.sh - cut a new release.
#
#   tools/release.sh 0.2.0
#
# What it does:
#   1. writes the version into VERSION
#   2. promotes the [Unreleased] section of CHANGELOG.md to "[<version>] - <today>"
#   3. commits, creates the annotated tag v<version>, pushes main + tag
#   4. the GitHub "Release" workflow then publishes the release page
#
# So: put your changes under "## [Unreleased]" in CHANGELOG.md first, then run this.
#
set -euo pipefail

ver="${1:-}"
if [ -z "$ver" ]; then
    echo "usage: tools/release.sh <version>   (e.g. tools/release.sh 0.2.0)" >&2
    exit 2
fi
ver="${ver#v}"                            # accept both 0.2.0 and v0.2.0
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if ! [[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
    echo "error: '$ver' is not a semver-like version (MAJOR.MINOR.PATCH)" >&2
    exit 2
fi
if git rev-parse -q --verify "refs/tags/v$ver" >/dev/null; then
    echo "error: tag v$ver already exists" >&2
    exit 2
fi

today="$(date +%Y-%m-%d)"
echo "$ver" > VERSION

python3 - "$ver" "$today" <<'PY'
import sys
ver, today = sys.argv[1], sys.argv[2]
p = "CHANGELOG.md"
s = open(p, encoding="utf-8").read()
if f"## [{ver}]" in s:
    sys.exit(f"error: CHANGELOG.md already has a [{ver}] section")
if "## [Unreleased]" not in s:
    sys.exit("error: CHANGELOG.md has no '## [Unreleased]' section")
s = s.replace("## [Unreleased]", f"## [Unreleased]\n\n## [{ver}] - {today}", 1)
open(p, "w", encoding="utf-8").write(s)
print(f"CHANGELOG.md: promoted [Unreleased] -> [{ver}] - {today}")
PY

git add VERSION CHANGELOG.md
git commit -m "release v$ver"
git tag -a "v$ver" -m "sdcpp-studio v$ver"
git push origin main
git push origin "v$ver"
echo "✅ Pushed v$ver. The 'Release' GitHub Action will publish the release page."
