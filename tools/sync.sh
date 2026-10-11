#!/usr/bin/env bash
#
# tools/sync.sh - update the mirror/backup copy on a remote machine.
#
#   tools/sync.sh                 # pull main + tags
#   tools/sync.sh --submodules    # also init/update the C++ submodules
#
# Configure once in .env (gitignored) or the environment:
#   SYNC_REMOTE   ssh target, e.g. user@host or an ssh alias   (required)
#   SYNC_DIR      repo path on the remote machine              (required)
#
# Example .env:
#   SYNC_REMOTE=myserver@192.0.2.10
#   SYNC_DIR=/srv/docker/sdcpp-studio
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# load .env defaults without overriding already-set variables
if [ -f "$root/.env" ]; then
    while IFS='=' read -r k v; do
        case "$k" in ''|\#*) continue ;; esac
        k="$(printf '%s' "$k" | tr -d '[:space:]')"
        [ -z "$k" ] && continue
        if [ -z "${!k:-}" ]; then
            v="${v%\"}"; v="${v#\"}"
            export "$k=$v"
        fi
    done < "$root/.env"
fi

: "${SYNC_REMOTE:?set SYNC_REMOTE (ssh target) in .env or environment}"
: "${SYNC_DIR:?set SYNC_DIR (remote repo path) in .env or environment}"

subs=0
[ "${1:-}" = "--submodules" ] && subs=1

remote_cmd="cd '$SYNC_DIR' && git fetch --tags origin && git pull --ff-only origin main"
if [ "$subs" = "1" ]; then
    remote_cmd="$remote_cmd && git submodule update --init --recursive"
fi
remote_cmd="$remote_cmd && echo '--- now ---' && git log --oneline -1 && echo \"VERSION=\$(cat VERSION 2>/dev/null || echo '?')\" && echo \"TAGS=\$(git tag | tr '\\n' ' ')\""

echo "Syncing ${SYNC_REMOTE}:${SYNC_DIR} ..."
ssh "$SYNC_REMOTE" "$remote_cmd"
