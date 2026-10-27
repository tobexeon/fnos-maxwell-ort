#!/usr/bin/env bash
#
# Copy the four runtime artifacts out of the cmake build tree.
#
# Usage:
#   bash scripts/collect.sh [destination]     (default: /dist)

set -euo pipefail

OUT_DIR="${OUT_DIR:-/build/out}"
DEST="${1:-/dist}"

mkdir -p "${DEST}"

for f in \
    onnxruntime_pybind11_state.so \
    libonnxruntime_providers_cuda.so \
    libonnxruntime_providers_shared.so \
    libonnxruntime.so.1.19.0
do
    cp -av "${OUT_DIR}/Release/${f}" "${DEST}/"
done

echo
echo "==> artifacts in ${DEST}"
ls -lh "${DEST}"
echo
md5sum "${DEST}"/*.so*
