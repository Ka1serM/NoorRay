#include "MaterialHit.slang"

// Surfaces whose material has no compiled shader: a neutral grey dielectric.
struct DefaultMaterial : IMaterial
{
    static surfaceshader evaluate(MaterialGeometry geometry, uint* parameters)
    {
        return mx_surface(mx_layer_bsdf(
                mx_dielectric_bsdf(1.0, float3(1.0), 1.5, float2(0.25), geometry.normal, 0),
                mx_oren_nayar_diffuse_bsdf(1.0, float3(0.8), 0.0, geometry.normal, false)),
            float3(0.0), 1.0);
    }
}

[shader("closesthit")]
void main(inout RealtimeHitPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    materialClosestHit<DefaultMaterial>(payload, attributes.barycentrics);
}

[shader("anyhit")]
void anyHit(RealtimeHitPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    materialAnyHit<DefaultMaterial>(payload, attributes.barycentrics);
}

[shader("anyhit")]
void shadowAnyHit(inout RealtimeShadowPayload payload,
    in BuiltInTriangleIntersectionAttributes attributes)
{
    materialShadowAnyHit<DefaultMaterial>(payload, attributes.barycentrics);
}
