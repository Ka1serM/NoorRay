#include "RaytracingAbi.slang"

// Ray type 1 is used only for a sampled mesh-light visibility query. No
// material emits until this renderer evaluates MaterialX shaders, so there
// are no mesh lights and the query never finds emission.
[shader("closesthit")]
void main(inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    payload.hit = 0u;
    payload.emissionSpectrum = float4(0.0);
}
