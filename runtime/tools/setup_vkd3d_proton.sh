#!/bin/sh
# Builds vkd3d-proton so the DX12 code runs on any Vulkan driver, Mesa lavapipe included.
# Needs meson, ninja, glslangValidator and widl (mingw-w64-tools); --windows also needs MinGW-w64.
#   runtime/tools/setup_vkd3d_proton.sh <dir> [--windows]
#     <dir>/install/{include,lib}   Linux build, for RDNU_VKD3D_DIR
#     <dir>/win64/d3d12*.dll        Windows build, for run_wine_tests.sh
# Run the tests with VKD3D_SHADER_MODEL=6_6. Lavapipe runs 16-bit shaders but does not preserve
# fp16 denormals, so vkd3d would hide native 16-bit ops; the build here reports them anyway.
set -e
dir=$(mkdir -p "${1:?usage: setup_vkd3d_proton.sh <dir> [--windows]}" && cd "$1" && pwd)
cd "$dir"
[ -d vkd3d-proton ] || git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/HansKristian-Work/vkd3d-proton.git
cd vkd3d-proton
sed -i 's/Native16BitShaderOpsSupported = d3d12_device_supports_16bit_shader_ops(device, false)/Native16BitShaderOpsSupported = d3d12_device_supports_16bit_shader_ops(device, true)/' libs/vkd3d/device.c
[ -d build.64 ] || meson setup --buildtype release --cross-file build-widl.txt --prefix "$dir/install" build.64
ninja -C build.64 install
[ "${2:-}" = --windows ] || exit 0
if [ ! -d build.win64 ]; then
    args=""
    # older MinGW-w64 headers lack it
    grep -qs PATHCCH_NONE /usr/x86_64-w64-mingw32/include/pathcch.h || args="-Dc_args=-DPATHCCH_NONE=0"
    meson setup --buildtype release --cross-file build-win64.txt $args build.win64
fi
ninja -C build.win64
mkdir -p "$dir/win64"
cp build.win64/libs/d3d12/d3d12.dll build.win64/libs/d3d12core/d3d12core.dll "$dir/win64/"
