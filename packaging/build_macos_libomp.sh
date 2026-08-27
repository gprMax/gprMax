#!/usr/bin/env bash
set -euo pipefail

# Homebrew bottles are built for the host macOS release and can therefore
# raise the minimum version required by a repaired wheel. Build the standalone
# LLVM OpenMP runtime under cibuildwheel's deployment target instead.
readonly LLVM_VERSION="21.1.8"
readonly RELEASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"
readonly OPENMP_ARCHIVE="openmp-${LLVM_VERSION}.src.tar.xz"
readonly OPENMP_SHA256="856b023748b41ac7b2c83fd8e9f765ff48a4df2fe6777d2811ef7c7ed8f2f977"
readonly CMAKE_ARCHIVE="cmake-${LLVM_VERSION}.src.tar.xz"
readonly CMAKE_SHA256="85735f20fd8c81ecb0a09abb0c267018475420e93b65050cc5b7634eab744de9"
readonly PREFIX="${GPRMAX_LIBOMP_PREFIX:-/tmp/gprmax-libomp}"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/gprmax-libomp.XXXXXX")"
trap 'rm -rf "${workdir}"' EXIT

curl --fail --location --retry 5 --output "${workdir}/${OPENMP_ARCHIVE}" \
    "${RELEASE_URL}/${OPENMP_ARCHIVE}"
curl --fail --location --retry 5 --output "${workdir}/${CMAKE_ARCHIVE}" \
    "${RELEASE_URL}/${CMAKE_ARCHIVE}"
printf '%s  %s\n' "${OPENMP_SHA256}" "${workdir}/${OPENMP_ARCHIVE}" | shasum -a 256 --check
printf '%s  %s\n' "${CMAKE_SHA256}" "${workdir}/${CMAKE_ARCHIVE}" | shasum -a 256 --check
tar -xf "${workdir}/${OPENMP_ARCHIVE}" -C "${workdir}"
tar -xf "${workdir}/${CMAKE_ARCHIVE}" -C "${workdir}"
mv "${workdir}/cmake-${LLVM_VERSION}.src" "${workdir}/cmake"

# The repaired wheel redistributes libomp, so ship the complete upstream
# licence alongside the bundled library.
mkdir -p gprMax/licenses
cp "${workdir}/openmp-${LLVM_VERSION}.src/LICENSE.TXT" \
    gprMax/licenses/LLVM-OpenMP-LICENSE.txt

cmake \
    -S "${workdir}/openmp-${LLVM_VERSION}.src" \
    -B "${workdir}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:?}" \
    -DLIBOMP_ENABLE_SHARED=ON \
    -DLIBOMP_ENABLE_STATIC=OFF \
    -DLIBOMP_FORTRAN_MODULES=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF
cmake --build "${workdir}/build" --parallel 2
cmake --install "${workdir}/build"

test -f "${PREFIX}/include/omp.h"
test -f "${PREFIX}/lib/libomp.dylib"
