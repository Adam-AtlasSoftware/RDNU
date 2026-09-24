// rdnu_embedded.h - data rdnu_shaderc builds into the runtime: DXIL per shader with the
// resources it really uses (from DXC's binding table), and the model files.
#pragma once

#include <cstddef>
#include <cstdint>

namespace rdnu
{

enum class BindingKind : uint8_t { Cbv, SrvTexture, UavTexture, SrvBuffer, UavBuffer, Sampler };

struct ShaderBinding
{
    const char* name;
    BindingKind kind;
    uint32_t    slot;  // register number
};

struct ShaderBlob
{
    const char*          name;
    const unsigned char* data;
    size_t               size;
    const ShaderBinding* bindings;
    uint32_t             bindingCount;
};

struct EmbeddedFile
{
    const char*          name;
    const unsigned char* data;
    size_t               size;
};

// Defined in the generated rdnu_embedded.cpp; nullptr when not built in.
const ShaderBlob*   FindShader(const char* name);
const EmbeddedFile* FindModelFile(const char* name);  // nss.rdnm, nss_w8.bin

}  // namespace rdnu
