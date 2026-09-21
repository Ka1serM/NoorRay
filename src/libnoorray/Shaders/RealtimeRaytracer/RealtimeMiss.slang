#include "../Raytracer/ByteBuffer.slang"
#include "../Raytracer/RaytracingAbi.slang"

// Environment and analytic light shapes are resolved by the ray-generation
// stage, which knows whether light sampling already accounted for them.
[shader("miss")]
void main(inout RealtimeRayPayload payload)
{
    payload.hit = 0u;
    payload.t = 1.0e30;
}
