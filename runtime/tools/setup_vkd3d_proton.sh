#!/bin/sh
# Builds vkd3d-proton natively so the DX12 code runs on Linux (any Vulkan driver, Mesa lavapipe
# included). Needs meson, ninja, glslangValidator and widl (mingw-w64-tools).
#   runtime/tools/setup_vkd3d_proton.sh <dir>   ->  <dir>/install/{include,lib}
# Run the tests with VKD3D_SHADER_MODEL=6_6. Lavapipe runs 16-bit shaders but does not preserve
# fp16 denormals, so vkd3d would hide native 16-bit ops; the build here reports them anyway.
set -e
dir=${1:?usage: setup_vkd3d_proton.sh <dir>}
mkdir -p "$dir"
cd "$dir"
[ -d vkd3d-proton ] || git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/HansKristian-Work/vkd3d-proton.git
cd vkd3d-proton
sed -i 's/Native16BitShaderOpsSupported = d3d12_device_supports_16bit_shader_ops(device, false)/Native16BitShaderOpsSupported = d3d12_device_supports_16bit_shader_ops(device, true)/' libs/vkd3d/device.c
[ -d build.64 ] || meson setup --buildtype release --cross-file build-widl.txt --prefix "$dir/install" build.64
ninja -C build.64 install
