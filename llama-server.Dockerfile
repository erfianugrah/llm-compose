# Blackwell-optimized llama.cpp build
# CUDA 12.8 + sm_120 native kernels for RTX 5090
# See: https://github.com/ggml-org/llama.cpp/pull/13360

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ARG LLAMA_CPP_VERSION=b8799

# Build dependencies
# - libssl-dev: TLS backend for cpp-httplib (used by -hf HuggingFace downloader)
# - libcurl4-openssl-dev is NOT needed (llama.cpp replaced libcurl with cpp-httplib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential curl ca-certificates \
      libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone llama.cpp (separate layer for caching)
RUN git clone --depth 1 --branch ${LLAMA_CPP_VERSION} \
      https://github.com/ggml-org/llama.cpp.git

WORKDIR /build/llama.cpp

# CMake configure + compile
# --allow-shlib-undefined: lets the linker accept unresolved libcuda.so symbols at
# build time. The real libcuda.so.1 is injected by the NVIDIA container runtime.
# This is the same approach used by the official llama.cpp .devops/cuda.Dockerfile.
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="120" \
      -DGGML_CUDA_FORCE_CUBLAS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined

RUN cmake --build build --config Release -j$(nproc) --target llama-server

# Collect shared libraries (.so files) that llama-server depends on.
# llama.cpp builds with BUILD_SHARED_LIBS=ON by default on Linux, producing
# libggml-base.so, libggml-cuda.so, libggml-cpu.so, libllama.so, libmtmd.so, etc.
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

# ---------------------------------------------------------------------------
# Runtime stage — CUDA runtime libs + server binary + shared libs
# nvidia/cuda:runtime includes libcublas and all CUDA shared libraries needed
# for inference. The base image would NOT work (only has libcudart).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# curl: healthcheck + HuggingFace downloads (llama-server uses built-in httplib)
# libssl3 / libgomp1: runtime deps for cpp-httplib TLS and OpenMP
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy shared libraries first, then the binary
COPY --from=build /app/lib/ /usr/local/lib/
RUN ldconfig
COPY --from=build /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Default model storage
VOLUME /models

EXPOSE 8080

ENTRYPOINT ["llama-server"]
