#include <d3dcompiler.h>
#include <iostream>

#pragma comment(lib, "d3dcompiler.lib")

// Minimal toolchain smoke test: compile one compute shader via d3dcompiler.
// Usage: test_compile [path-to-hlsl]  (default: runtime/shaders/rdg_upsample.hlsl, relative to CWD)
// NOTE: cs_5_0 cannot compile the SM6.x WMMA shaders; this only sanity-checks that d3dcompiler is wired up.
int wmain(int argc, wchar_t** argv) {
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;

    const wchar_t* shaderPath = (argc > 1) ? argv[1] : L"runtime/shaders/rdg_upsample.hlsl";
    HRESULT hr = D3DCompileFromFile(shaderPath, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "RDG_Upsample_CS", "cs_5_0", flags, 0, &shaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cout << "Shader Compile Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return 1;
    }
    std::cout << "Compile successful!\n";
    return 0;
}