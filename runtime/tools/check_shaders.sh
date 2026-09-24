#!/usr/bin/env sh
# Compile-check every harness kernel with DXC (SM 6.2, the engine's profile). Needs `dxc` on PATH
# or DXC=/path/to/dxc; Linux builds: github.com/microsoft/DirectXShaderCompiler/releases.
# The wmma_* kernels need the AMD AGS header (-I <FidelityFX-SDK>/Kits/FidelityFX/api/internal/dx12/AmdExtD3D)
# and the AMD dxc build, so they are skipped unless AGS_INC is set.
set -u
DXC=${DXC:-dxc}
DIR=$(dirname "$0")/../harness/shaders
fail=0
for f in "$DIR"/*.hlsl; do
    name=$(basename "$f" .hlsl)
    case "$name" in wmma_*) [ -n "${AGS_INC:-}" ] || continue; inc="-I$AGS_INC";; *) inc="";; esac
    entry=$(grep -o '^void [A-Za-z0-9_]*_CS' "$f" | head -1 | cut -d' ' -f2)
    if out=$("$DXC" -T cs_6_2 -E "$entry" -enable-16bit-types $inc "$f" -Fo /dev/null 2>&1); then
        printf '%-20s OK\n' "$name"
    else
        printf '%-20s FAIL\n%s\n' "$name" "$out"; fail=1
    fi
done
exit $fail
