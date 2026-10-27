# Build environment for ONNX Runtime 1.19.0 with NVIDIA Maxwell (sm_50) support.
#
# Base image matches the CUDA / cuDNN pair used by fnOS ai_manager:
#   CUDA 12.6 + cuDNN 9 + Ubuntu 22.04
#
# Build:
#   docker build -t fnos-maxwell-ort:1.19.0 .
#
# Run (mount a work dir, compile inside):
#   docker run --rm -it --gpus all \
#     -v "$PWD/build:/build" -v "$PWD/dist:/dist" \
#     --shm-size=2g -m 16g \
#     fnos-maxwell-ort:1.19.0 bash
#
# NOTE: compiling the CUDA execution provider peaks well above 8 GB RSS.
#       Use at least 16 GB of memory for the container, otherwise nvcc will be OOM-killed.

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- base toolchain + Python 3.11 (deadsnakes) ------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl git wget \
        build-essential ninja-build ccache pkg-config \
        libssl-dev zlib1g-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ---- CMake (ORT 1.19 requires >= 3.26) --------------------------------------
ARG CMAKE_VERSION=3.28.3
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
      | tar -xz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/

# ---- Eigen 3.4 (header-only, used by ORT) -----------------------------------
ARG EIGEN_VERSION=3.4.0
RUN git clone --depth 1 --branch ${EIGEN_VERSION} https://gitlab.com/libeigen/eigen.git /opt/eigen

# ---- Python venv used to drive tools/ci_build/build.py ----------------------
RUN python3.11 -m venv /opt/py311 \
    && /opt/py311/bin/pip install --no-cache-dir -U pip wheel \
    && /opt/py311/bin/pip install --no-cache-dir numpy packaging flatbuffers sympy

# ---- ccache: keeps incremental rebuilds fast --------------------------------
ENV CCACHE_DIR=/build/.ccache
ENV CCACHE_MAXSIZE=10G

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/opt/py311/bin:${PATH}

WORKDIR /build
