// rdnu_amd_ext.cpp - see rdnu_amd_ext.h. Builds against the AmdExtD3D headers shipped in the
// FidelityFX SDK (api/internal/dx12/AmdExtD3D) when RDNU_HAVE_AMDEXT is defined.
#include "rdnu_amd_ext.h"

#if defined(_WIN32) && defined(RDNU_HAVE_AMDEXT)
#include <AmdExtD3D.h>
#include <AmdExtD3DDeviceApi.h>
#include <AmdExtD3DShaderIntrinsicsApi.h>

#ifdef __MINGW32__  // __uuidof needs the ids spelled out for mingw
__CRT_UUID_DECL(IAmdExtD3DFactory, 0x014937ec, 0x9288, 0x446f, 0xa9, 0xac, 0xd7, 0x5a, 0x8e, 0x3a, 0x98, 0x4f)
__CRT_UUID_DECL(IAmdExtD3DDevice8, 0xf714e11a, 0xb54e, 0x4e0f, 0xab, 0xc5, 0xdf, 0x58, 0xb1, 0x81, 0x33, 0xd1)
__CRT_UUID_DECL(IAmdExtD3DShaderIntrinsics, 0xba019d53, 0xccab, 0x4cbd, 0xb5, 0x6a, 0x72, 0x30, 0xed, 0x43, 0x30, 0xad)
#endif

namespace rdnu
{
namespace
{
template <typename T>
struct Ref
{
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
};
}  // namespace

bool EnableAmdWaveMatrixInt8(ID3D12Device* device)
{
    HMODULE driver = GetModuleHandleW(L"amdxc64.dll");
    if (!driver)
        return false;
    auto create = reinterpret_cast<PFNAmdExtD3DCreateInterface>(reinterpret_cast<void (*)()>(GetProcAddress(driver, "AmdExtD3DCreateInterface")));
    Ref<IAmdExtD3DFactory> factory;
    if (!create || FAILED(create(device, IID_PPV_ARGS(&factory.p))))
        return false;

    Ref<IAmdExtD3DDevice8> ext;
    if (FAILED(factory.p->CreateInterface(device, IID_PPV_ARGS(&ext.p))))
        return false;
    using T = AmdExtWaveMatrixProperties::Type;
    AmdExtWaveMatrixProperties props[32];
    size_t count = sizeof(props) / sizeof(props[0]);
    if (FAILED(ext.p->GetWaveMatrixProperties(&count, props)))
        return false;
    bool int8 = false;
    for (size_t i = 0; i < count && i < sizeof(props) / sizeof(props[0]); ++i)
    {
        const AmdExtWaveMatrixProperties& p = props[i];
        int8 |= p.mSize == 16 && p.nSize == 16 && p.kSize == 16 && p.aType == T::sint8 && p.bType == T::sint8 &&
                p.cType == T::sint32 && p.resultType == T::sint32 && !p.saturatingAccumulation;
    }
    if (!int8)
        return false;

    Ref<IAmdExtD3DShaderIntrinsics> intrinsics;
    return SUCCEEDED(factory.p->CreateInterface(device, IID_PPV_ARGS(&intrinsics.p))) &&
           SUCCEEDED(intrinsics.p->CheckSupport(AmdExtD3DShaderIntrinsicsSupport_WaveMatrix)) && SUCCEEDED(intrinsics.p->Enable());
}

}  // namespace rdnu
#else
namespace rdnu
{
bool EnableAmdWaveMatrixInt8(ID3D12Device*) { return false; }
}  // namespace rdnu
#endif
