#!/bin/bash
set -e

exec 9<"$0"
flock 9

source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null || true

CMAKE_QUIET=()
BUILD_QUIET=()
if [ -n "$CI" ]; then
    # Drop CMake info chatter and Make's per-target progress; keep warnings,
    # errors, and compiler stderr intact.
    CMAKE_QUIET=(--log-level=WARNING -Wno-dev)
    BUILD_QUIET=(-- -s)
fi

chronic cmake -B build-local "${CMAKE_QUIET[@]}" \
    -DGGML_NATIVE=OFF -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

chronic cmake --build build-local --config Release -j"$(nproc)" "${BUILD_QUIET[@]}"
chronic ccache -s

echo "Build completed"
