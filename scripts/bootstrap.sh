#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

hloc_revision=c13273bd0ecc2917a35910fd843712a1c6243193
hloc_url=https://github.com/cvg/Hierarchical-Localization.git
if ! command -v uv >/dev/null 2>&1; then
    printf 'Install uv before running this script.\n' >&2
    exit 1
fi

if [ ! -e external/hloc ]; then
    mkdir -p external
    git clone --quiet --no-checkout "$hloc_url" external/hloc
    git -C external/hloc checkout --quiet --detach "$hloc_revision"
fi
if [ "$(git -C external/hloc rev-parse HEAD)" != "$hloc_revision" ] ||
   [ "$(git -C external/hloc remote get-url origin)" != "$hloc_url" ] ||
   [ -n "$(git -C external/hloc status --porcelain --untracked-files=normal)" ]; then
    printf 'external/hloc must be a clean checkout of %s from %s; refusing to overwrite it.\n' \
        "$hloc_revision" "$hloc_url" >&2
    exit 1
fi
# hloc's SuperPoint wrapper imports this source tree and its bundled weights.
git -C external/hloc submodule update --init --depth 1 third_party/SuperGluePretrainedNetwork
uv sync --locked --python python3.10
