#!/usr/bin/env bash
# Create the private toolchain that util/build_exl3_ext.sh builds exllamav3_ext with: CUDA 12.8.1 from NVIDIA's redist
# archives plus gcc-11 unpacked from distro packages. No root needed; nothing is installed into the system or into
# the Python environment, and deleting the toolchain directory removes everything.
#
# usage: util/setup_toolchain_for_exl3_ext.sh
# env:   PYTHON      python with the torch the extension will be built against (default: the active virtualenv's).
#                    Its torch must be a CUDA 12.x build; its nvidia-* pip packages supply the cuBLAS, cuSPARSE,
#                    cuSOLVER and cuRAND headers, so they match what torch runs with
#        TOOLCHAINS  where to create it (default ~/toolchains): cuda-12.8.1/, gcc-11/, and dl/ with the downloads
#                    (reused on a rerun when the checksums match)
#        GCC_DEBS    directory holding the gcc-11 .debs already (skips apt-get download; see below)
#
# Needs curl, tar, xz, sha256sum, dpkg and system binutils (as, ld). gcc-11 comes from `apt-get download`, which needs
# no root but needs the release to still ship gcc-11 (Ubuntu 22.04 to 26.04 do). Otherwise download these six debs
# for a release that has them into GCC_DEBS and rerun: gcc-11 g++-11 cpp-11 gcc-11-base libgcc-11-dev libstdc++-11-dev.
#
# Why these versions: nvcc's major version must match torch's CUDA (12 for cu12x wheels), and 12.8.1 is what the
# official cu128 exllamav3 wheels are built with. CUDA 12.8 rejects gcc newer than 14; gcc 11 matches the wheels' CI.
# glibc 2.41+ declares sinpi/cospi/rsqrt noexcept, which CUDA 12.8's crt/math_functions.h conflicts with; the setup
# patches those declarations the way CUDA 12.9 does (declarations only, generated code is unchanged; the original is
# kept as math_functions.h.orig). The patch is guarded on the glibc version, so it is harmless on older systems.
set -euo pipefail
T=${TOOLCHAINS:-$HOME/toolchains}
PY=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
CUDA=$T/cuda-12.8.1; GCC=$T/gcc-11; DL=$T/dl
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
# sha256 and path below $REDIST, from redistrib_12.8.1.json
ARCHIVES="
9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3 cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68 cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz
0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz
7755524026d30e2c211d793392166517e379724d19eb9e8e352693f7d5fa0c95 cuda_nvtx/linux-x86_64/cuda_nvtx-linux-x86_64-12.8.90-archive.tar.xz
f97b3779e909caa5edf56606432ec1dd3173a5d3d503ef39f367467dbdb006b7 cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-12.8.90-archive.tar.xz
dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz
8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz
"
DEBS="gcc-11 g++-11 cpp-11 gcc-11-base libgcc-11-dev libstdc++-11-dev"
die() { echo "error: $*" >&2; exit 1; }

# --- checks
[ "$(uname -m)" = x86_64 ] || die "x86_64 only"
for c in curl tar xz sha256sum dpkg as ld; do command -v $c > /dev/null || die "missing command: $c"; done
[ -n "$PY" ] || die "set PYTHON to the python whose torch the extension must match (or activate its venv)"
tcuda=$("$PY" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null) || die "$PY cannot import torch"
[ "${tcuda%%.*}" = 12 ] || die "torch is built for CUDA '${tcuda:-none}'; this toolchain (CUDA 12.8.1) needs a CUDA 12.x torch"
NV=$("$PY" -c "import nvidia; print(list(nvidia.__path__)[0])" 2>/dev/null) || die "$PY has no nvidia-* pip packages"
for l in cublas cusparse cusolver curand; do [ -d "$NV/$l/include" ] || die "no $NV/$l/include (pip package nvidia-$l-cu12)"; done
for d in "$CUDA" "$GCC"; do [ ! -e "$d" ] || die "$d exists; delete it first or set TOOLCHAINS elsewhere"; done
mkdir -p "$DL/deb"
echo "toolchain -> $T (torch CUDA $tcuda, headers from $NV)"

# --- CUDA 12.8.1: download, verify, merge the archives into one tree
stage=$(mktemp -d "$T/.cuda-stage.XXXX"); gstage=$(mktemp -d "$T/.gcc-stage.XXXX")
trap 'rm -rf "$stage" "$gstage"' EXIT
echo "$ARCHIVES" | while read -r sum path; do
    [ -n "$sum" ] || continue
    f=$DL/$(basename "$path")
    if ! { [ -f "$f" ] && echo "$sum  $f" | sha256sum -c --status > /dev/null 2>&1; }; then
        echo "  download $(basename "$path")"
        curl -fsSL --retry 3 -o "$f.part" "$REDIST/$path" && mv "$f.part" "$f"
        echo "$sum  $f" | sha256sum -c --status || die "checksum mismatch: $f"
    fi
    tar -xJf "$f" -C "$stage" --strip-components=1
done
rm -f "$stage/LICENSE"
mkdir -p "$stage/targets/x86_64-linux"                       # the layout nvcc.profile expects
ln -s ../../include "$stage/targets/x86_64-linux/include"
ln -s ../../lib "$stage/targets/x86_64-linux/lib"
ln -s lib "$stage/lib64"
for l in cublas cusparse cusolver curand; do cp -r "$NV/$l/include/." "$stage/include/"; done
rm -rf "$stage/include/__init__.py" "$stage/include/__pycache__"

# --- glibc >= 2.41 noexcept declarations in crt/math_functions.h
mf=$stage/include/crt/math_functions.h; cp "$mf" "$mf.orig"
"$PY" - "$mf" << 'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
guard = "#define __MATH_FUNCTIONS_H__\n"
block = ("/* local patch (setup_toolchain_for_exl3_ext.sh): glibc >= 2.41 declares sinpi/cospi/rsqrt noexcept; match it, as CUDA 12.9 does */\n"
         "#if defined(__cplusplus) && defined(__GLIBC__) && defined(__GLIBC_PREREQ)\n#if __GLIBC_PREREQ(2, 41)\n"
         "#define __NV_LOCAL_GLIBC_NOEXCEPT noexcept(true)\n#endif\n#endif\n"
         "#ifndef __NV_LOCAL_GLIBC_NOEXCEPT\n#define __NV_LOCAL_GLIBC_NOEXCEPT\n#endif\n")
assert s.count(guard) == 1, "include guard not found"
s = s.replace(guard, guard + block)
for fn, ty in (("rsqrt", "double"), ("rsqrtf", "float"), ("sinpi", "double"), ("sinpif", "float"), ("cospi", "double"), ("cospif", "float")):
    pat = re.compile(r"(^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ %s +%s\(%s x\));$" % (ty, fn, ty), re.M)
    s, n = pat.subn(r"\1 __NV_LOCAL_GLIBC_NOEXCEPT;", s)
    assert n == 1, f"declaration of {fn} not found"
open(p, "w").write(s)
EOF

# --- gcc-11 from distro packages
DEBDIR=${GCC_DEBS:-$DL/deb}
missing=""; for p in $DEBS; do ls "$DEBDIR"/${p}_*_amd64.deb > /dev/null 2>&1 || missing="$missing $p"; done
if [ -n "$missing" ]; then
    [ -z "${GCC_DEBS:-}" ] || die "GCC_DEBS=$GCC_DEBS lacks:$missing"
    command -v apt-get > /dev/null || die "no apt-get: put the gcc-11 debs ($DEBS) in a directory and set GCC_DEBS"
    echo "  apt-get download$missing"
    (cd "$DEBDIR" && apt-get download $missing) || die "apt-get could not download$missing; see the header for GCC_DEBS"
fi
for p in $DEBS; do dpkg -x "$(ls -t "$DEBDIR"/${p}_*_amd64.deb | head -1)" "$gstage"; done
# The runtime-library links point into a usr/lib/x86_64-linux-gnu the toolchain does not have: use the system copies
# (sanitizer libraries stay dangling when the system lacks them; nothing here uses them)
for l in "$gstage"/usr/lib/gcc/x86_64-linux-gnu/11/*.so; do
    [ -L "$l" ] || continue
    t=$(readlink "$l"); [ "${t#../../../x86_64-linux-gnu/}" != "$t" ] || continue
    sys=/usr/lib/x86_64-linux-gnu/$(basename "$t"); [ -e "$sys" ] && ln -sfn "$sys" "$l"
done

# --- smoke tests: host compiler, then nvcc with that host compiler against the patched headers
tmp=$(mktemp -d); trap 'rm -rf "$stage" "$gstage" "$tmp"' EXIT
printf '#include <cmath>\n#include <string>\nint main() { return (int) std::sqrt((double) std::string("ok").size()) - 1; }\n' > "$tmp/t.cpp"
"$gstage/usr/bin/g++-11" -O1 "$tmp/t.cpp" -o "$tmp/t" && "$tmp/t" \
    || die "gcc-11 cannot build a test program (missing libisl/libmpc/libmpfr/libgmp or binutils?); see ldd $gstage/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus"
printf '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cmath>\n__global__ void k(float* x) { x[threadIdx.x] = sinpif(x[threadIdx.x]) + rsqrtf(2.0f); }\n' > "$tmp/t.cu"
"$stage/bin/nvcc" -ccbin "$gstage/usr/bin/g++-11" -std=c++17 -arch=sm_86 -c "$tmp/t.cu" -o "$tmp/t.o" \
    || die "nvcc test compile failed"

chmod 755 "$stage" "$gstage"                                  # mktemp -d made them 700
mv "$stage" "$CUDA"; mv "$gstage" "$GCC"
echo "OK: $CUDA ($("$CUDA/bin/nvcc" --version | grep -oE 'V[0-9.]+')), $GCC ($("$GCC/usr/bin/g++-11" --version | head -1))"
echo "build: PYTHON=$PY util/build_exl3_ext.sh"
