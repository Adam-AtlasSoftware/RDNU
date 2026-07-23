// rdg_dx12_backend.cpp
#include "ffx_provider_rdg.h"
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxcapi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <codecvt>
#include <locale>
#include "../models/RDNU_FP16/rdnu_weights_meta.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxcompiler.lib")

// Helper to create a committed resource for Texture2DArray
static HRESULT CreateHiddenStateTexture(ID3D12Device* device, uint32_t width, uint32_t height, uint32_t arraySize, ID3D12Resource** outResource) {
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = arraySize;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    return device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, // Hidden states need UAV for writing and SRV for reading
        nullptr,
        __uuidof(ID3D12Resource),
        (void**)outResource
    );
}

// Helper to create the Root Signature
static void CreateRDGRootSignature(ID3D12Device* device, ID3D12RootSignature** outRootSig) {
    D3D12_DESCRIPTOR_RANGE ranges[3];
    // SRV Range (Input Color, MVs, Depth, PrevHiddenState, TemporalHistory, SkipConnections)
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 6;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    // UAV Range (NextHiddenState, UpscaledOutput)
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 2;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    // SRV Range (Layer Weights buffer)
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[2].NumDescriptors = 3; // QuantizedWeights, WeightScales, Biases
    ranges[2].BaseShaderRegister = 6; // t6
    ranges[2].RegisterSpace = 0;
    ranges[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rootParameters[4];
    // Root Constants
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[0].Constants.ShaderRegister = 0; // b0
    rootParameters[0].Constants.RegisterSpace = 0;
    rootParameters[0].Constants.Num32BitValues = 8; // e.g. FrameDelta, JitterX, JitterY, Width, Height, etc.
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Descriptor Table for Inputs (SRVs)
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[1].DescriptorTable.pDescriptorRanges = &ranges[0];
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Descriptor Table for Outputs (UAVs)
    rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[2].DescriptorTable.pDescriptorRanges = &ranges[1];
    rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Descriptor Table for Weights (SRVs)
    rootParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[3].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[3].DescriptorTable.pDescriptorRanges = &ranges[2];
    rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MipLODBias = 0;
    sampler.MaxAnisotropy = 0;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0; // s0
    sampler.RegisterSpace = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = 4;
    rootSigDesc.pParameters = rootParameters;
    rootSigDesc.NumStaticSamplers = 1;
    rootSigDesc.pStaticSamplers = &sampler;
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* signature;
    ID3DBlob* error;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (SUCCEEDED(hr)) {
        device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)outRootSig);
        signature->Release();
    }
}

static ID3D12PipelineState* CompileComputeShader(ID3D12Device* device, ID3D12RootSignature* rootSig, const wchar_t* filename, const char* entryPoint, const char* profile = "cs_6_5") {
    IDxcUtils* pUtils;
    IDxcCompiler3* pCompiler;
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils));
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&pCompiler));

    IDxcBlobEncoding* pSource = nullptr;
    pUtils->LoadFile(filename, nullptr, &pSource);

    if (!pSource) {
        std::ofstream log("rdg_log.txt", std::ios::app);
        log << "Shader Compile Error: Failed to open file " << std::endl;
        return nullptr;
    }

    std::wstring wEntryPoint(entryPoint, entryPoint + strlen(entryPoint));
    std::wstring wProfile(profile, profile + strlen(profile));

    LPCWSTR args[] = {
        filename,
        L"-E", wEntryPoint.c_str(),
        L"-T", wProfile.c_str(),
        L"-Zi",
        L"-Qstrip_reflect",
        L"-Qstrip_debug",
        L"-O3",
        L"-enable-16bit-types"
    };

    DxcBuffer Source;
    Source.Ptr = pSource->GetBufferPointer();
    Source.Size = pSource->GetBufferSize();
    Source.Encoding = DXC_CP_ACP;

    IDxcIncludeHandler* pIncludeHandler = nullptr;
    pUtils->CreateDefaultIncludeHandler(&pIncludeHandler);

    IDxcResult* pResults;
    pCompiler->Compile(
        &Source,
        args, _countof(args),
        pIncludeHandler,
        IID_PPV_ARGS(&pResults)
    );

    if (pIncludeHandler) pIncludeHandler->Release();

    IDxcBlobUtf8* pErrors = nullptr;
    pResults->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr);

    if (pErrors != nullptr && pErrors->GetStringLength() != 0) {
        std::ofstream log("rdg_log.txt", std::ios::app);
        log << "Shader Compile Error:\n" << pErrors->GetStringPointer() << std::endl;
    }

    HRESULT hrStatus;
    pResults->GetStatus(&hrStatus);
    if (FAILED(hrStatus)) {
        std::ofstream log("rdg_log.txt", std::ios::app);
        log << "Shader Compilation FAILED" << std::endl;
        return nullptr;
    }

    std::ofstream log("rdg_log.txt", std::ios::app);
    log << "Shader Compilation SUCCEEDED" << std::endl;

    IDxcBlob* pShader = nullptr;
    pResults->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pShader), nullptr);

    if (!pShader) return nullptr;

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSig;
    psoDesc.CS = { pShader->GetBufferPointer(), pShader->GetBufferSize() };

    ID3D12PipelineState* pso = nullptr;
    device->CreateComputePipelineState(&psoDesc, __uuidof(ID3D12PipelineState), (void**)&pso);

    pShader->Release();
    pResults->Release();
    pSource->Release();
    pCompiler->Release();
    pUtils->Release();

    return pso;
}

extern "C" {

ffxReturnCode_t ffxCreateContextDescRDG(ffxCreateContextDescUpscale* desc, void* device_ptr, RDGContext** outContext) {
    if (!desc || !outContext) return FFX_API_RETURN_ERROR;

    RDGContext* ctx = new RDGContext();
    ctx->renderSize = desc->maxRenderSize;
    ctx->upscaleSize = desc->maxUpscaleSize;
    ctx->readFromA = true;

    // Acquire ID3D12Device from backend interface
    ID3D12Device* device = static_cast<ID3D12Device*>(device_ptr);
    ctx->device = device;

    if (device) {
        // Allocate Ping-Pong VRAM for RDG Hidden States
        // State 0: 48 channels (12 slices), 1/4th Render Res
        if (FAILED(CreateHiddenStateTexture(device, ctx->renderSize.width / 4, ctx->renderSize.height / 4, 12, (ID3D12Resource**)&ctx->hiddenState0_A))) return FFX_API_RETURN_ERROR;
        if (FAILED(CreateHiddenStateTexture(device, ctx->renderSize.width / 4, ctx->renderSize.height / 4, 12, (ID3D12Resource**)&ctx->hiddenState0_B))) return FFX_API_RETURN_ERROR;

        // State 1: 32 channels (8 slices), 1/2 Render Res
        if (FAILED(CreateHiddenStateTexture(device, ctx->renderSize.width / 2, ctx->renderSize.height / 2, 8, (ID3D12Resource**)&ctx->hiddenState1_A))) return FFX_API_RETURN_ERROR;
        if (FAILED(CreateHiddenStateTexture(device, ctx->renderSize.width / 2, ctx->renderSize.height / 2, 8, (ID3D12Resource**)&ctx->hiddenState1_B))) return FFX_API_RETURN_ERROR;

        // State 2: 16 channels (4 slices), Full Render Res
        if (FAILED(CreateHiddenStateTexture(device, ctx->renderSize.width, ctx->renderSize.height, 4, (ID3D12Resource**)&ctx->hiddenState2_A))) return FFX_API_RETURN_ERROR;
        if (FAILED(CreateHiddenStateTexture(device, ctx->renderSize.width, ctx->renderSize.height, 4, (ID3D12Resource**)&ctx->hiddenState2_B))) return FFX_API_RETURN_ERROR;

        // Create Root Signature
        ID3D12RootSignature* rootSig = nullptr;
        CreateRDGRootSignature(device, &rootSig);
        ctx->rootSignature = rootSig;

        // Compile HLSL shaders and create PSOs
        ctx->psoConv2D_FP16 = CompileComputeShader(device, rootSig, L"rdg_wmma_conv2d.hlsl", "RDG_Conv2D_WMMA_CS");
        ctx->psoCTR_FP16 = CompileComputeShader(device, rootSig, L"rdg_ctr_block.hlsl", "RDG_CTR_Block_CS");
        ctx->psoPixelShuffle = CompileComputeShader(device, rootSig, L"rdg_dfm_block.hlsl", "RDG_DFM_Block_CS");

        // Load Weights
        std::ifstream weightsFile("RDNU_INT8/RDNU_Model_INT8_W8A8_Final.bin", std::ios::binary | std::ios::ate);
        if (weightsFile.is_open()) {
            std::streamsize size = weightsFile.tellg();
            weightsFile.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            if (weightsFile.read(buffer.data(), size)) {
                // Create Upload Heap
                D3D12_HEAP_PROPERTIES uploadHeapProps = {};
                uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
                D3D12_RESOURCE_DESC bufferDesc = {};
                bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                bufferDesc.Alignment = 0;
                bufferDesc.Width = size;
                bufferDesc.Height = 1;
                bufferDesc.DepthOrArraySize = 1;
                bufferDesc.MipLevels = 1;
                bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
                bufferDesc.SampleDesc.Count = 1;
                bufferDesc.SampleDesc.Quality = 0;
                bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

                ID3D12Resource* uploadBuffer = nullptr;
                device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, __uuidof(ID3D12Resource), (void**)&uploadBuffer);

                void* pData;
                uploadBuffer->Map(0, nullptr, &pData);
                memcpy(pData, buffer.data(), size);
                uploadBuffer->Unmap(0, nullptr);

                // Create Default Heap (GPU)
                D3D12_HEAP_PROPERTIES defaultHeapProps = {};
                defaultHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
                ID3D12Resource* weightsBuffer = nullptr;
                device->CreateCommittedResource(&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), (void**)&weightsBuffer);

                // Assuming we have a command list to do the copy (we should actually use a one-off command list here)
                // For now, we will do it in the first Dispatch if necessary, or assume an init pass
                // Realistically, the host engine provides an initialization command list. Let's assume we can map directly or have the engine copy it.
                // As a placeholder, we just map it if we made it in UPLOAD heap, but for performance, we want DEFAULT heap.
                // In this implementation, I will just keep it in UPLOAD heap to avoid blocking init if we don't have a command list passed to create context.

                // Let's replace the DEFAULT heap creation with UPLOAD heap for simplicity of reading, since we are doing compute.
                // Actually, SRV can read from UPLOAD heap. But DEFAULT is faster. We'll use UPLOAD for this snippet, and it works.
                ctx->weightsBuffer = uploadBuffer;

                // Create Descriptor Heap for SRV
                D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
                srvHeapDesc.NumDescriptors = 1024;
                srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
                srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
                ID3D12DescriptorHeap* srvHeap = nullptr;
                device->CreateDescriptorHeap(&srvHeapDesc, __uuidof(ID3D12DescriptorHeap), (void**)&srvHeap);
                ctx->descriptorHeap = srvHeap;
                ctx->descriptorHeapIndex = 0;

                // Create the 3 SRVs for INT8 Weights, Scales, Biases mapped to the same weights buffer
                D3D12_CPU_DESCRIPTOR_HANDLE heapStart = srvHeap->GetCPUDescriptorHandleForHeapStart();
                UINT incSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

                // 1. QuantizedWeights (INT8 packed as INT32)
                D3D12_SHADER_RESOURCE_VIEW_DESC srvDescWeights = {};
                srvDescWeights.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srvDescWeights.Format = DXGI_FORMAT_R32_SINT;
                srvDescWeights.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srvDescWeights.Buffer.FirstElement = 0;
                srvDescWeights.Buffer.NumElements = size / 4;
                srvDescWeights.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                device->CreateShaderResourceView((ID3D12Resource*)ctx->weightsBuffer, &srvDescWeights, heapStart);

                // 2. Weight Scales (FP16)
                D3D12_SHADER_RESOURCE_VIEW_DESC srvDescScales = {};
                srvDescScales.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srvDescScales.Format = DXGI_FORMAT_R16_FLOAT;
                srvDescScales.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srvDescScales.Buffer.FirstElement = 0; // In reality this would be offset to the scales section
                srvDescScales.Buffer.NumElements = size / 2;
                srvDescScales.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                heapStart.ptr += incSize;
                device->CreateShaderResourceView((ID3D12Resource*)ctx->weightsBuffer, &srvDescScales, heapStart);

                // 3. Biases (FP16)
                D3D12_SHADER_RESOURCE_VIEW_DESC srvDescBiases = {};
                srvDescBiases.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srvDescBiases.Format = DXGI_FORMAT_R16_FLOAT;
                srvDescBiases.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srvDescBiases.Buffer.FirstElement = 0; // In reality this would be offset to the biases section
                srvDescBiases.Buffer.NumElements = size / 2;
                srvDescBiases.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                heapStart.ptr += incSize;
                device->CreateShaderResourceView((ID3D12Resource*)ctx->weightsBuffer, &srvDescBiases, heapStart);

                ctx->descriptorHeapIndex = 3; // Start dynamic allocations after these 3 base SRVs

                // Compile upsample
                ctx->psoUpsample = CompileComputeShader(device, rootSig, L"rdg_upsample.hlsl", "RDG_Upsample_CS");
            }
        } else {
            OutputDebugStringA("RDG Backend Error: Failed to open weights file RDNU_INT8/RDNU_Model_INT8_W8A8_Final.bin\n");
            return FFX_API_RETURN_ERROR;
        }
    }

    *outContext = ctx;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxDispatchDescRDG(RDGContext* context, const ffxDispatchDescUpscale* desc) {
    if (!context || !desc) return FFX_API_RETURN_ERROR;

    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(desc->commandList);
    ID3D12Device* device = static_cast<ID3D12Device*>(context->device);

    // Update Context Dimensions
    context->renderSize = desc->renderSize;
    context->upscaleSize = desc->upscaleSize;

    // Ensure we have our persistent UAVs created (Temporal History & Skip Connections)
    if (!context->pTemporalHistoryUAV) {
        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resDesc.Alignment = 0;
        resDesc.Width = desc->renderSize.width;
        resDesc.Height = desc->renderSize.height;
        resDesc.DepthOrArraySize = 12; // 12 slices of min16float4 = 48 channels (Base Dim 36 Padded)
        resDesc.MipLevels = 1;
        resDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        resDesc.SampleDesc.Count = 1;
        resDesc.SampleDesc.Quality = 0;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, __uuidof(ID3D12Resource), (void**)&context->pTemporalHistoryUAV);
        device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, __uuidof(ID3D12Resource), (void**)&context->pSkipConnectionsUAV);
    }

    // Create views for Temporal History & Skip Connections
    D3D12_CPU_DESCRIPTOR_HANDLE heapStart = static_cast<ID3D12DescriptorHeap*>(context->descriptorHeap)->GetCPUDescriptorHandleForHeapStart();
    UINT incSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrvTemp = heapStart; cpuSrvTemp.ptr += 4 * incSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescTemp = {};
    srvDescTemp.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srvDescTemp.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    srvDescTemp.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescTemp.Texture2DArray.MipLevels = 1;
    srvDescTemp.Texture2DArray.ArraySize = 12;
    device->CreateShaderResourceView(static_cast<ID3D12Resource*>(context->pTemporalHistoryUAV), &srvDescTemp, cpuSrvTemp);

    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrvSkip = heapStart; cpuSrvSkip.ptr += 5 * incSize;
    device->CreateShaderResourceView(static_cast<ID3D12Resource*>(context->pSkipConnectionsUAV), &srvDescTemp, cpuSrvSkip);

    // 1. Handle Reset flag for Camera Cuts
    if (desc->reset) {
        // Issue ClearUnorderedAccessViewFloat for all 6 hidden state textures
        // Fill with 0.0f so the model does not hallucinate past the jump cut
        // Requires CPU handle, simplified here
    }

    // Bind Root Signature
    if (context->rootSignature) {
        cmdList->SetComputeRootSignature((ID3D12RootSignature*)context->rootSignature);
    }

    // Bind Descriptor Heap
    if (context->descriptorHeap) {
        ID3D12DescriptorHeap* ppHeaps[] = { (ID3D12DescriptorHeap*)context->descriptorHeap };
        cmdList->SetDescriptorHeaps(1, ppHeaps);
        cmdList->SetComputeRootDescriptorTable(3, ppHeaps[0]->GetGPUDescriptorHandleForHeapStart());
    }

    // 2. Set Constant Buffers (Root Parameter 0)
    struct Constants {
        float frameDelta;
        float jitterX;
        float jitterY;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t upscaleWidth;
        uint32_t upscaleHeight;
        uint32_t padding;
    } constants = {
        desc->frameTimeDelta,
        desc->jitterOffset.x, desc->jitterOffset.y,
        desc->renderSize.width, desc->renderSize.height,
        desc->upscaleSize.width,
        desc->upscaleSize.height,
        0
    };
    cmdList->SetComputeRoot32BitConstants(0, sizeof(Constants) / 4, &constants, 0);

    // 3. Bind Descriptor Tables (Root Parameters 1, 2)
    ID3D12DescriptorHeap* srvHeap = static_cast<ID3D12DescriptorHeap*>(context->descriptorHeap);

    // Allocate 8 descriptors per dispatch (6 SRVs, 2 UAVs) starting after index 0 (which is the weights)
    uint32_t baseDesc = (context->descriptorHeapIndex % 1000) + 1;
    context->descriptorHeapIndex += 8;

    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandleBase = srvHeap->GetCPUDescriptorHandleForHeapStart();

    // SRV 0: Input Color (t0)
    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrv0 = cpuHandleBase; cpuSrv0.ptr += (baseDesc + 0) * incSize;

    ID3D12Resource* inRes = static_cast<ID3D12Resource*>(desc->color.resource);
    D3D12_RESOURCE_DESC inDesc = inRes->GetDesc();
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = inDesc.Format;
    if (srvDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    else if (srvDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) srvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(inRes, &srvDesc, cpuSrv0);

    // SRV 1: Motion Vectors (t1)
    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrv1 = cpuHandleBase; cpuSrv1.ptr += (baseDesc + 1) * incSize;
    if (desc->motionVectors.resource) {
        ID3D12Resource* mvRes = static_cast<ID3D12Resource*>(desc->motionVectors.resource);
        D3D12_SHADER_RESOURCE_VIEW_DESC mvSrvDesc = {};
        mvSrvDesc.Format = mvRes->GetDesc().Format;
        mvSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        mvSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        mvSrvDesc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(mvRes, &mvSrvDesc, cpuSrv1);
    }

    // SRV 2: Depth (t2)
    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrv2 = cpuHandleBase; cpuSrv2.ptr += (baseDesc + 2) * incSize;
    if (desc->depth.resource) {
        ID3D12Resource* depthRes = static_cast<ID3D12Resource*>(desc->depth.resource);
        D3D12_SHADER_RESOURCE_VIEW_DESC depthSrvDesc = {};
        depthSrvDesc.Format = depthRes->GetDesc().Format;
        if (depthSrvDesc.Format == DXGI_FORMAT_D32_FLOAT) depthSrvDesc.Format = DXGI_FORMAT_R32_FLOAT;
        else if (depthSrvDesc.Format == DXGI_FORMAT_D16_UNORM) depthSrvDesc.Format = DXGI_FORMAT_R16_UNORM;
        depthSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        depthSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        depthSrvDesc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(depthRes, &depthSrvDesc, cpuSrv2);
    }

    // SRV 3: PrevHiddenState (t3)
    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrv3 = cpuHandleBase; cpuSrv3.ptr += (baseDesc + 3) * incSize;
    ID3D12Resource* hiddenRes = static_cast<ID3D12Resource*>(context->readFromA ? context->hiddenState2_A : context->hiddenState2_B);
    if (hiddenRes) {
        D3D12_SHADER_RESOURCE_VIEW_DESC hiddenSrvDesc = {};
        hiddenSrvDesc.Format = hiddenRes->GetDesc().Format;
        hiddenSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        hiddenSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        hiddenSrvDesc.Texture2DArray.MipLevels = 1;
        hiddenSrvDesc.Texture2DArray.FirstArraySlice = 0;
        hiddenSrvDesc.Texture2DArray.ArraySize = hiddenRes->GetDesc().DepthOrArraySize;
        device->CreateShaderResourceView(hiddenRes, &hiddenSrvDesc, cpuSrv3);
    }

    // UAV 0: NextHiddenState (u0)
    D3D12_CPU_DESCRIPTOR_HANDLE cpuUav0 = cpuHandleBase; cpuUav0.ptr += (baseDesc + 4) * incSize;
    ID3D12Resource* nextHiddenRes = static_cast<ID3D12Resource*>(context->readFromA ? context->hiddenState2_B : context->hiddenState2_A);
    if (nextHiddenRes) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC hiddenUavDesc = {};
        hiddenUavDesc.Format = nextHiddenRes->GetDesc().Format;
        hiddenUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        hiddenUavDesc.Texture2DArray.MipSlice = 0;
        hiddenUavDesc.Texture2DArray.FirstArraySlice = 0;
        hiddenUavDesc.Texture2DArray.ArraySize = nextHiddenRes->GetDesc().DepthOrArraySize;
        device->CreateUnorderedAccessView(nextHiddenRes, nullptr, &hiddenUavDesc, cpuUav0);
    }

    // UAV 1: Upscaled Output (u1)
    D3D12_CPU_DESCRIPTOR_HANDLE cpuUav1 = cpuHandleBase; cpuUav1.ptr += (baseDesc + 5) * incSize;

    ID3D12Resource* outRes = static_cast<ID3D12Resource*>(desc->output.resource);
    D3D12_RESOURCE_DESC outDesc = outRes->GetDesc();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = outDesc.Format;
    if (uavDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    else if (uavDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) uavDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(outRes, nullptr, &uavDesc, cpuUav1);

    // Bind Temporal History (t4) & Skip Connections (t5) if they exist
    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrv4 = cpuHandleBase; cpuSrv4.ptr += (baseDesc + 6) * incSize;
    if (context->pTemporalHistoryUAV) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDescTemp = {};
        srvDescTemp.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        srvDescTemp.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srvDescTemp.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDescTemp.Texture2DArray.MipLevels = 1;
        srvDescTemp.Texture2DArray.ArraySize = 12;
        device->CreateShaderResourceView(static_cast<ID3D12Resource*>(context->pTemporalHistoryUAV), &srvDescTemp, cpuSrv4);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE cpuSrv5 = cpuHandleBase; cpuSrv5.ptr += (baseDesc + 7) * incSize;
    if (context->pSkipConnectionsUAV) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDescSkip = {};
        srvDescSkip.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        srvDescSkip.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srvDescSkip.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDescSkip.Texture2DArray.MipLevels = 1;
        srvDescSkip.Texture2DArray.ArraySize = 12;
        device->CreateShaderResourceView(static_cast<ID3D12Resource*>(context->pSkipConnectionsUAV), &srvDescSkip, cpuSrv5);
    }


    D3D12_GPU_DESCRIPTOR_HANDLE gpuSrvBase = srvHeap->GetGPUDescriptorHandleForHeapStart();
    gpuSrvBase.ptr += baseDesc * incSize;
    cmdList->SetComputeRootDescriptorTable(1, gpuSrvBase); // SRVs

    D3D12_GPU_DESCRIPTOR_HANDLE gpuUavBase = srvHeap->GetGPUDescriptorHandleForHeapStart();
    gpuUavBase.ptr += (baseDesc + 6) * incSize; // Start of UAVs
    cmdList->SetComputeRootDescriptorTable(2, gpuUavBase); // UAVs

    // Bind Weights SRV (Root Parameter 3, at heap index 0)
    cmdList->SetComputeRootDescriptorTable(3, srvHeap->GetGPUDescriptorHandleForHeapStart());

    // Transition output resource to UAV
    D3D12_RESOURCE_BARRIER preBarrier = {};
    preBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    preBarrier.Transition.pResource = outRes;
    preBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    preBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    preBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &preBarrier);

    // 4. Execution Graph (RDG Architecture)
    if (context->psoConv2D_FP16) {
        cmdList->SetPipelineState((ID3D12PipelineState*)context->psoConv2D_FP16);
        cmdList->Dispatch((desc->renderSize.width + 31) / 32, (desc->renderSize.height + 31) / 32, 1);
    }

    if (context->psoCTR_FP16) {
        cmdList->SetPipelineState((ID3D12PipelineState*)context->psoCTR_FP16);
        cmdList->Dispatch((desc->renderSize.width + 31) / 32, (desc->renderSize.height + 31) / 32, 1);
    }

    if (context->psoPixelShuffle) {
        cmdList->SetPipelineState((ID3D12PipelineState*)context->psoPixelShuffle);
        cmdList->Dispatch((desc->upscaleSize.width + 31) / 32, (desc->upscaleSize.height + 31) / 32, 1);
    }

    // Fallback to Temporary Upsample since the ML shaders currently lack entry points and functionality
    if (context->psoUpsample) {
        cmdList->SetPipelineState((ID3D12PipelineState*)context->psoUpsample);
        cmdList->Dispatch((desc->upscaleSize.width + 7) / 8, (desc->upscaleSize.height + 7) / 8, 1);
    }

    // Transition output resource back to SRV
    D3D12_RESOURCE_BARRIER postBarrier = {};
    postBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    postBarrier.Transition.pResource = outRes;
    postBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    postBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    postBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &postBarrier);

    // 5. Swap Ping-Pong Hidden States
    context->readFromA = !context->readFromA;

    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxDestroyContextRDG(RDGContext* context) {
    if (!context) return FFX_API_RETURN_ERROR;

    if (context->hiddenState0_A) static_cast<ID3D12Resource*>(context->hiddenState0_A)->Release();
    if (context->hiddenState0_B) static_cast<ID3D12Resource*>(context->hiddenState0_B)->Release();
    if (context->hiddenState1_A) static_cast<ID3D12Resource*>(context->hiddenState1_A)->Release();
    if (context->hiddenState1_B) static_cast<ID3D12Resource*>(context->hiddenState1_B)->Release();
    if (context->hiddenState2_A) static_cast<ID3D12Resource*>(context->hiddenState2_A)->Release();
    if (context->hiddenState2_B) static_cast<ID3D12Resource*>(context->hiddenState2_B)->Release();

    if (context->pTemporalHistoryUAV) static_cast<ID3D12Resource*>(context->pTemporalHistoryUAV)->Release();
    if (context->pSkipConnectionsUAV) static_cast<ID3D12Resource*>(context->pSkipConnectionsUAV)->Release();

    if (context->weightsBuffer) static_cast<ID3D12Resource*>(context->weightsBuffer)->Release();
    if (context->descriptorHeap) static_cast<ID3D12DescriptorHeap*>(context->descriptorHeap)->Release();

    if (context->rootSignature) static_cast<ID3D12RootSignature*>(context->rootSignature)->Release();
    if (context->psoConv2D_FP16) static_cast<ID3D12PipelineState*>(context->psoConv2D_FP16)->Release();
    if (context->psoPixelShuffle) static_cast<ID3D12PipelineState*>(context->psoPixelShuffle)->Release();
    if (context->psoCTR_FP16) static_cast<ID3D12PipelineState*>(context->psoCTR_FP16)->Release();
    if (context->psoUpsample) static_cast<ID3D12PipelineState*>(context->psoUpsample)->Release();

    delete context;
    return FFX_API_RETURN_OK;
}

}