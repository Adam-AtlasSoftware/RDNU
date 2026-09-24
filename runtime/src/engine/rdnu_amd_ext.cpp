// rdnu_amd_ext.cpp - see rdnu_amd_ext.h. Builds against the AmdExtD3D headers shipped in the
// FidelityFX SDK (api/internal/dx12/AmdExtD3D) when RDNU_HAVE_AMDEXT is defined.
#include "rdnu_amd_ext.h"

#if defined(_WIN32) && defined(RDNU_HAVE_AMDEXT)
#include <AmdExtD3D.h>
#include <AmdExtD3DDeviceApi.h>
#include <AmdExtD3DShaderIntrinsicsApi.h>

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
    auto create = reinterpret_cast<PFNAmdExtD3DCreateInterface>(GetProcAddress(driver, "AmdExtD3DCreateInterface"));
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
