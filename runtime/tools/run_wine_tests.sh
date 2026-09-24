#!/bin/sh
# Runs the Windows build under Wine on Mesa lavapipe, the way games load it:
#   amd_fidelityfx_dx12.dll            FSR 3.1: the DLL's own exports
#   amd_fidelityfx_loader_dx12.dll     FidelityFX SDK 2: AMD's signed loader calling RDNU as
#                                      amd_fidelityfx_upscaler_dx12.dll through its provider object
# with AMD's upscaler beside it as the _original, so FSR 3.1 runs forwarded for comparison.
#
#   runtime/tools/run_wine_tests.sh <mingw build dir> <vkd3d dir> <FidelityFX SDK> [golden dir]
#
# <vkd3d dir> from setup_vkd3d_proton.sh --windows. Needs wine (WINE=... to override) and
# xvfb-run: Wine's DXGI wants a display, the same one for every Wine process.
set -e
[ -n "${DISPLAY:-}" ] || exec xvfb-run -a sh "$0" "$@"
build=$(cd "${1:?usage: run_wine_tests.sh <mingw build dir> <vkd3d dir> <FidelityFX SDK> [golden dir]}" && pwd)
vkd3d=$(cd "${2:?}" && pwd)
sdk=$(cd "${3:?}" && pwd)
root=$(cd "$(dirname "$0")/.." && pwd)
golden=$(cd "${4:-$root/tools/golden}" && pwd)
WINE=${WINE:-wine}
run="$build/wine"
mkdir -p "$run"
cp "$build"/runtime/*.exe "$build"/runtime/*.dll "$vkd3d"/win64/d3d12*.dll "$run/"
cp "$sdk/Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll" "$run/"
cp "$sdk/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll" "$run/amd_fidelityfx_upscaler_dx12_original.dll"
export WINEPREFIX="$run/prefix" WINEDEBUG=-all WINEDLLOVERRIDES="d3d12,d3d12core=n" VKD3D_SHADER_MODEL=6_6 VKD3D_DEBUG=err
cd "$run"
[ -d prefix ] || "$WINE" wineboot -i
winpath() { echo "Z:$1" | tr / '\\'; }
g=$(winpath "$golden")
"$WINE" test_engine_core.exe "$(winpath "$root/models/NSS_INT8")" "$g" | tail -1
"$WINE" test_upscale_map.exe | tail -1
"$WINE" rdnu_prod.exe "$g" | tail -1
"$WINE" test_nss_dx12.exe "$g" --frames 8 | tail -1
for dll in amd_fidelityfx_dx12.dll amd_fidelityfx_loader_dx12.dll; do
    echo "== $dll"
    "$WINE" test_fsr_api.exe "$g" --dll $dll
done
