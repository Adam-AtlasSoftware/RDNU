#!/bin/sh
# Builds vkd3d-proton natively so the DX12 code runs on Linux (any Vulkan driver, Mesa lavapipe
# included). Needs meson, ninja, glslangValidator and widl (mingw-w64-tools).
#   runtime/tools/setup_vkd3d_proton.sh <dir>   ->  <dir>/install/{include,lib}
set -e
dir=${1:?usage: setup_vkd3d_proton.sh <dir>}
mkdir -p "$dir"
cd "$dir"
[ -d vkd3d-proton ] || git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/HansKristian-Work/vkd3d-proton.git
cd vkd3d-proton
[ -d build.64 ] || meson setup --buildtype release --cross-file build-widl.txt --prefix "$dir/install" build.64
ninja -C build.64 install
