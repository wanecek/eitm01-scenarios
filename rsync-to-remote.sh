#!/usr/bin/env bash
REMOTE_HOST="DO" # Aliased
REMOTE_PATH="/root/cibus/scenarios"

while inotifywait -r -e modify,create,delete,move --exclude='.git/' .; do
  # v: verbose
  # h: ???
  # a: Sync recursively & preserve symbolic links, modification times, owner, etc
  # z: archive files that have not, but can be, archived
  # P: --progress (progress-bar) and --partial (resume interrupted transfers)
  rsync -vhazP \
    . "$REMOTE_HOST:$REMOTE_PATH" \
    --delete \
    --include='**.gitignore' \
    --exclude='/.git' \
    --exclude='rsync-to-dev.sh' \
    --filter='dir-merge,- .gitignore'
done
