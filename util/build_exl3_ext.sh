#!/usr/bin/env bash
# Build exllamav3_ext in place with a private CUDA + gcc toolchain, without installing anything into the Python
# environment or the system. Objects go to <checkout>/build/, the .so to <checkout>/ (both gitignored); a source
# checkout used as the package (e.g. tabbyAPI's developer.exllamav3_path) then loads that .so.
#
# usage: util/build_exl3_ext.sh [exllamav3 checkout]      (default: the checkout this script is in)
#
# env:   PYTHON      interpreter whose torch the extension is built against (only read, nothing is installed into
#                    it); default: the active virtualenv's python. Required otherwise
#        TOOLCHAINS  toolchain directory (default ~/toolchains), holding:
#                      <TOOLKIT>/          a CUDA toolkit whose major version matches torch's (e.g. the NVIDIA redist
#                                          archives of 12.8.1 for a cu128 torch), with targets/x86_64-linux/{include,lib}
#                                          and lib64 linked to include/lib, and the cuBLAS/cuSPARSE/cuSOLVER/cuRAND
#                                          headers torch was built with
#                      gcc-11/usr/bin/     gcc-11 and g++-11 (e.g. Ubuntu debs unpacked with dpkg -x); CUDA 12.8
#                                          rejects newer host compilers
#                    With glibc >= 2.41, <TOOLKIT>/include/crt/math_functions.h needs sinpi/cospi/rsqrt(f) declared
#                    noexcept, as CUDA 12.9 does (declarations only, generated code is unchanged)
#        TOOLKIT     toolkit directory name inside TOOLCHAINS (default cuda-12.8.1)
#        BUILD_TAG   separate object dir build/temp...-<tag> and a copy of the .so in $TOOLCHAINS/builds/<tag>/
#        TORCH_CUDA_ARCH_LIST (default "8.6;8.9;12.0"), MAX_JOBS (default 20)
#
# Build only while no model is loaded: the nvcc jobs need tens of GB of RAM. Full build ~5 min, incremental after.
# undo:  rm <checkout>/exllamav3_ext*.so   (the package then loads the installed wheel's extension)
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
CHECKOUT=$(readlink -f "${1:-$HERE/..}")
T=$(readlink -f "${TOOLCHAINS:-$HOME/toolchains}")
PY=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
[ -n "$PY" ] || { echo "set PYTHON to the python whose torch the extension must match (or activate its venv)" >&2; exit 1; }
[ -f "$CHECKOUT/setup.py" ] || { echo "no setup.py in $CHECKOUT" >&2; exit 1; }
"$PY" -c "import torch" 2>/dev/null || { echo "$PY cannot import torch" >&2; exit 1; }

TK=$T/${TOOLKIT:-cuda-12.8.1}; [ -x "$TK/bin/nvcc" ] || { echo "no toolkit $TK" >&2; exit 1; }
GCC=$T/gcc-11/usr/bin; [ -x "$GCC/g++-11" ] || { echo "no $GCC/g++-11" >&2; exit 1; }
export CUDA_HOME=$TK CUDA_PATH=$TK
export CC=$GCC/gcc-11 CXX=$GCC/g++-11 CUDAHOSTCXX=$GCC/g++-11
export PATH=$CUDA_HOME/bin:$(dirname "$PY"):$PATH          # nvcc, ninja
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6;8.9;12.0}
export MAX_JOBS=${MAX_JOBS:-20}
unset PYTORCH_NVCC EXLLAMA_NOCOMPILE

cd "$CHECKOUT"
mkdir -p build
echo "build $(git rev-parse --short HEAD) (+$(git status --porcelain -- exllamav3/exllamav3_ext setup.py | wc -l) changed files)" \
     "| ${TOOLKIT:-cuda-12.8.1} tag ${BUILD_TAG:-default} | nvcc $(nvcc --version | grep -oE 'V[0-9]+\.[0-9.]+') | $($CXX --version | head -1)" \
     "| archs $TORCH_CUDA_ARCH_LIST | jobs $MAX_JOBS | $(date +%T)"
start=$(date +%s); touch build/.stamp
TEMP=build/temp.linux-x86_64-cpython-312${BUILD_TAG:+-$BUILD_TAG}; LOG=build/build${BUILD_TAG:+-$BUILD_TAG}.log
set +e; "$PY" setup.py build_ext --inplace --build-temp "$TEMP" > $LOG 2>&1; rc=$?; set -e     # progress: tail -f build/build.log
grep -E "error:|error [A-Z]|Error [0-9]|FAILED:" $LOG | head -20 || true
so=$(ls -t "$CHECKOUT"/exllamav3_ext*.so 2>/dev/null | head -1 || true)
if [ "$rc" != 0 ] || [ -z "$so" ] || ! [ "$so" -nt build/.stamp ]; then
    echo "BUILD FAILED (rc=$rc), see $CHECKOUT/$LOG" >&2; exit 1
fi
[ -n "${BUILD_TAG:-}" ] && mkdir -p "$T/builds/$BUILD_TAG" && cp "$so" "$T/builds/$BUILD_TAG/"
echo "BUILD OK in $(( $(date +%s) - start ))s: $so ($(du -h "$so" | cut -f1))"
