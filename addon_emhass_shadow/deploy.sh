#!/bin/sh
# Deploy the EMHASS Shadow add-on to a Home Assistant OS host over SSH.
#
# Refreshes the deploy-time copies (models are kept site-private, src/ mirrors
# the repo's src/emhass), uploads via a tar stream (the host has no scp/sftp),
# then reloads the store and rebuilds the add-on.
#
# Usage:
#   ./deploy.sh [user@host] [ssh_identity_file]
# Defaults: timka@192.168.5.160 with ../data/forecast_training/ha_host_ssh
#
# Run from the repository root or from the add-on directory.

set -eu

ADDON_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname "$ADDON_DIR")"
TARGET="${1:-timka@192.168.5.160}"
KEY="${2:-$REPO_ROOT/data/forecast_training/ha_host_ssh}"

echo ">> syncing src/emhass into the add-on build context"
rsync -a --delete "$REPO_ROOT/src/emhass/" "$ADDON_DIR/src/emhass/"

if [ ! -d "$ADDON_DIR/models/load_quantiles_v1" ]; then
  echo "!! $ADDON_DIR/models is missing (gitignored, site-private)." >&2
  echo "!! The Docker build needs it; on an existing host the copy in" >&2
  echo "!! /addons/emhass_shadow/models is reused as-is." >&2
fi

echo ">> uploading to $TARGET"
COPYFILE_DISABLE=1 tar czf - --exclude='__pycache__' \
  -C "$ADDON_DIR" config.yaml Dockerfile run.sh app src |
  ssh -i "$KEY" -o BatchMode=yes "$TARGET" '
    rm -rf /tmp/emhass_shadow_up && mkdir -p /tmp/emhass_shadow_up &&
    tar xzf - -C /tmp/emhass_shadow_up &&
    sudo rm -rf /addons/emhass_shadow/app/__pycache__ &&
    sudo tar cf - -C /tmp/emhass_shadow_up . | sudo tar xf - -C /addons/emhass_shadow'

echo ">> reloading store and rebuilding the add-on"
ssh -i "$KEY" -o BatchMode=yes "$TARGET" '
  sudo docker exec hassio_cli ha store reload >/dev/null 2>&1 || true
  sudo docker exec hassio_cli ha apps rebuild local_emhass_shadow'

echo ">> done; check: ssh -i $KEY $TARGET 'sudo docker exec hassio_cli ha apps info local_emhass_shadow'"
