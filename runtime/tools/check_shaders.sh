#!/usr/bin/env sh
# Compile-check the shaders with DXC: dxc on PATH or DXC=/path/to/dxc (Linux builds:
# github.com/microsoft/DirectXShaderCompiler/releases). AGS_INC=<FidelityFX SDK>/Kits/FidelityFX/
# api/internal/dx12/AmdExtD3D adds the wave-matrix kernels.
#
#   harness/shaders            the reference engine's kernels, SM 6.2 (wave matrix 6.6)
#   shaders/net, shaders/nss   every network kernel and pass permutation the runtime embeds,
#                              through rdnu_shaderc (RDNU_SHADERC=path, built by CMake)
set -u
DXC=${DXC:-dxc}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
fail=0
for f in "$ROOT"/harness/shaders/*.hlsl; do
    name=$(basename "$f" .hlsl)
    case "$name" in wmma_*) [ -n "${AGS_INC:-}" ] || continue; inc="-I$AGS_INC"; sm=cs_6_6;; *) inc=""; sm=cs_6_2;; esac
    entry=$(grep -o '^void [A-Za-z0-9_]*_CS' "$f" | head -1 | cut -d' ' -f2)
    if out=$("$DXC" -T $sm -E "$entry" -enable-16bit-types $inc "$f" -Fo /dev/null 2>&1); then
        printf '%-24s OK\n' "$name"
    else
        printf '%-24s FAIL\n%s\n' "$name" "$out"; fail=1
    fi
done
if [ -n "${RDNU_SHADERC:-}" ]; then
    tmp=$(mktemp -d)
    if [ -n "${AGS_INC:-}" ]; then wmma="--amd-ext $AGS_INC"; else wmma="--no-wmma"; fi
    if "$RDNU_SHADERC" --dxc "$DXC" --shaders "$ROOT/shaders" --model "$ROOT/models/NSS_INT8" $wmma --out "$tmp/embedded.cpp"; then
        printf '%-24s OK\n' "runtime shaders"
    else
        fail=1
    fi
    rm -rf "$tmp"
else
    echo "RDNU_SHADERC not set: runtime shaders skipped"
fi
exit $fail
