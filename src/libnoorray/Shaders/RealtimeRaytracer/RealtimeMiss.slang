#include "HitPayload.slang"

// Environment and analytic light shapes are resolved by the ray-generation
// stage, which knows whether light sampling already accounted for them.
[shader("miss")]
void main(inout RealtimeHitPayload payload)
{
    payload.instance = RealtimeMissInstance;
    payload.t = 1.0e30;
}
