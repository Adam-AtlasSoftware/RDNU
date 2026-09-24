// nss_fill.hlsl - fills a raw buffer with one dword pattern; clears the activation arena to
// -128 codes (0x80808080) once, after which the conv kernels maintain every border themselves.
cbuffer FillConstants : register(b0)
{
    uint Quads;    // number of 16-byte stores
    uint Value;
    uint GroupsX;
};

RWByteAddressBuffer Output : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint i = (gid.y * GroupsX + gid.x) * 256 + gi;
    if (i < Quads)
        Output.Store4(i * 16, uint4(Value, Value, Value, Value));
}
