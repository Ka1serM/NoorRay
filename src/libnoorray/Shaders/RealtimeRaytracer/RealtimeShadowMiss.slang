#include "../Raytracer/RaytracingAbi.slang"

[shader("miss")]
void main(inout RealtimeShadowPayload payload)
{
    payload.occluded = 0u;
}
