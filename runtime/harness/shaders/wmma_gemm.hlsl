// wmma_gemm.hlsl - minimal RDNA3 WMMA probe: one 16x16x16 INT8 -> INT32 GEMM tile.
// C[16x16](int32) = A[16x16](int8) @ B[16x16](int8), one Wave32. Validates that the vendored
// AMD wave-matrix-capable dxc + the 7900 XTX actually run the AGS wave-matrix intrinsics, and
// pins the load layout convention before building the tiled INT8 conv.

#define AmdExtD3DShaderIntrinsics_EnableWaveMatrix
#include "AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"    // pulls in the AGS base UAV (u0, magic space)

ByteAddressBuffer    MatA : register(t0);   // 16x16 int8, row-major (256 bytes)
ByteAddressBuffer    MatB : register(t1);   // 16x16 int8, row-major
RWByteAddressBuffer   MatC : register(u1);   // 16x16 int32 (1024 bytes)  (u0 is the AGS mailbox)

[WaveSize(32)]            // RDNA3 WMMA requires Wave32 (SM 6.6)
[numthreads(32, 1, 1)]
void wmma_gemm_CS(uint3 gtid : SV_GroupThreadID)
{
    AmdWaveMatrixA<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> a;
    AmdWaveMatrixB<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> b;
    AmdWaveMatrixAccumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> c;

    a.Load(MatA, 0, 16, false);   // row-major, 16 bytes/row (16 int8)
    b.Load(MatB, 0, 16, false);
    c.Fill(0);
    c = AmdWaveMatrixMultiply(a, b, c);   // C = A*B + C
    c.Store(MatC, 0, 64, false);  // row-major, 64 bytes/row (16 int32)
}
