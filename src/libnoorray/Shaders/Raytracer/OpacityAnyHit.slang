#include "ByteBuffer.slang"
#include "RaytracingAbi.slang"
#include "SceneSurface.slang"
#include "Sampling.slang"

// Stochastic vertex-colour coverage is resolved during traversal so primary,
// continuation, analytic-shadow, and mesh-emission rays all see the same
// transparent geometry semantics.
[shader("anyhit")]
void main(inout RayPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    const Frame frame = *root.arguments;
    const RegularSurface surface = loadRegularSurface(frame.scene, InstanceID(),
        GeometryIndex(), PrimitiveIndex(), attributes.barycentrics);
    // A mixed-sided mesh disables facing culling for the whole TLAS instance.
    // Reject back faces here for its one-sided sections in every ray type.
    if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE
        && (frame.materials[surface.materialIndex]->flags & MaterialFlagOneSided) != 0u)
    {
        IgnoreHit();
        return;
    }
    const float blockProbability = saturate(surface.vertexColor.a);
    if (blockProbability >= 1.0)
        return;
    Rng rng;
    rng.state = hashCombine32(payload.seed,
        hashCombine32(InstanceID(), PrimitiveIndex() ^ (payload.depth * 0x9E3779B9u)));
    const bool ignore = rng.nextFloat() >= blockProbability;
    payload.seed = rng.state;
    if (ignore)
        IgnoreHit();
}
