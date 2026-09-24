// rdnu_shader_blobs.h - DXIL compiled offline by rdnu_shaderc, looked up by kernel name.
#pragma once

#include <cstddef>

namespace rdnu
{

struct ShaderBlob
{
    const char*          name;
    const unsigned char* data;
    size_t               size;
};

// Defined in the generated rdnu_shaders_dxil.cpp; nullptr when the name was not compiled.
const ShaderBlob* FindShader(const char* name);

}  // namespace rdnu
