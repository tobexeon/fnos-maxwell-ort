#!/usr/bin/env bash
#
# Clone ONNX Runtime v1.19.0, apply the Maxwell patches, and build.
# Intended to run inside the image built from this repo's Dockerfile.
#
# Usage:
#   bash scripts/build.sh
#
# Overridable env:
#   ORT_VERSION  git tag to build          (default: v1.19.0)
#   SRC_DIR      source checkout path      (default: /build/onnxruntime)
#   OUT_DIR      cmake build dir           (default: /build/out)
#   JOBS         parallel compile jobs     (default: nproc)

set -euo pipefail

ORT_VERSION="${ORT_VERSION:-v1.19.0}"
SRC_DIR="${SRC_DIR:-/build/onnxruntime}"
OUT_DIR="${OUT_DIR:-/build/out}"
JOBS="${JOBS:-$(nproc)}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"

echo "==> ONNX Runtime ${ORT_VERSION}"
echo "    source : ${SRC_DIR}"
echo "    build  : ${OUT_DIR}"
echo "    jobs   : ${JOBS}"
echo "    patches: ${PATCH_DIR}"

if [ ! -d "${SRC_DIR}/.git" ]; then
    echo "==> cloning onnxruntime ${ORT_VERSION}"
    git clone --depth 1 --branch "${ORT_VERSION}" \
        https://github.com/microsoft/onnxruntime.git "${SRC_DIR}"
fi

cd "${SRC_DIR}"

for p in "${PATCH_DIR}"/*.patch; do
    name="$(basename "$p")"
    if git apply --reverse --check "$p" >/dev/null 2>&1; then
        echo "==> ${name}: already applied, skipping"
        continue
    fi
    echo "==> applying ${name}"
    git apply --check "$p"
    git apply "$p"
done

echo "==> building (first run takes 1-3 h, incremental rebuild ~30 s with warm ccache)"
python3 tools/ci_build/build.py \
    --build_dir "${OUT_DIR}" \
    --config Release \
    --build_shared_lib \
    --build_wheel \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --cuda_version 12.6 \
    --use_preinstalled_eigen \
    --eigen_path /opt/eigen \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=50 \
    --cmake_extra_defines CMAKE_C_COMPILER_LAUNCHER=ccache \
    --cmake_extra_defines CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    --cmake_extra_defines CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    --compile_no_warning_as_error \
    --enable_pybind \
    --allow_running_as_root \
    --skip_tests \
    --parallel "${JOBS}" \
    --nvcc_threads 1

echo "==> build finished: ${OUT_DIR}/Release"
