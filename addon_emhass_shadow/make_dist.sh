#!/bin/sh
# Build a distributable EMHASS Shadow add-on archive for offline installs.
#
# Produces dist/emhass-shadow-<version>.tar.gz with a Home Assistant add-on
# repository layout (repository.yaml + the add-on folder), ready to copy into
# /addons on a HAOS host or to push as a GitHub add-on repository later.
#
# Site-private forecast models are EXCLUDED by default (they encode this
# home's consumption patterns); fresh installs bootstrap their own models.
# Pass --with-models to include them anyway.
#
# Run from the repository root or from the add-on directory.

set -eu

ADDON_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname "$ADDON_DIR")"
VERSION=$(grep -m1 '^version:' "$ADDON_DIR/config.yaml" | sed 's/.*"\(.*\)"/\1/')
WITH_MODELS=0
[ "${1:-}" = "--with-models" ] && WITH_MODELS=1

echo ">> syncing src/emhass into the add-on build context"
rsync -a --delete "$REPO_ROOT/src/emhass/" "$ADDON_DIR/src/emhass/"

STAGE="$REPO_ROOT/dist/.stage-emhass-shadow"
rm -rf "$STAGE"
mkdir -p "$STAGE/emhass_shadow" "$REPO_ROOT/dist"

cat > "$STAGE/repository.yaml" <<EOF
name: EMHASS Shadow
url: https://example.invalid/emhass-shadow
maintainer: local distribution
EOF

rsync -a --exclude='__pycache__' \
  "$ADDON_DIR/config.yaml" "$ADDON_DIR/Dockerfile" "$ADDON_DIR/run.sh" \
  "$ADDON_DIR/README.md" "$ADDON_DIR/app" "$ADDON_DIR/src" \
  "$STAGE/emhass_shadow/"

if [ "$WITH_MODELS" -eq 1 ]; then
  echo ">> including site-trained models (privacy: they encode this home's data)"
  rsync -a "$ADDON_DIR/models" "$STAGE/emhass_shadow/"
else
  # The Dockerfile COPY models/ needs the directory; fresh installs start
  # empty and bootstrap their own models from accrued history.
  mkdir -p "$STAGE/emhass_shadow/models"
fi

OUT="$REPO_ROOT/dist/emhass-shadow-$VERSION.tar.gz"
COPYFILE_DISABLE=1 tar czf "$OUT" -C "$STAGE" repository.yaml emhass_shadow
rm -rf "$STAGE"
echo ">> $OUT"
echo ">> install: unpack, copy emhass_shadow/ into /addons on the HAOS host,"
echo ">> then Add-on Store -> Check for updates -> Local add-ons -> Install"
