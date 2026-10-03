#include "../../Shared/LandscapeMesh.h"

[[vk::push_constant]] ConstantBuffer<LandscapeMeshRoot> root;

// Z is the original R << 8 | G texel value. Keep it exact until this pass,
// then apply Unreal's landscape vertex-factory height conversion.
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const LandscapeMeshArguments args = *root.arguments;
    if (id.x >= args.vertexCount) return;
    float3 point = args.positions[id.x];
    point.z = (point.z - 32768.0) / 128.0;
    args.positions[id.x] = point;
}
