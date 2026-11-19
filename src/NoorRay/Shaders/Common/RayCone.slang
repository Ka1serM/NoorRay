import Common.Math;
import Common.Frame;

//! Akenine-Möller et al. 2021: width is the cone diameter at the ray origin, spread the full cone angle in radians.
struct RayCone {
    float width;
    float spread;
};

RayCone makeRayCone(float width, float spread) {
    RayCone cone;
    cone.width = width;
    cone.spread = spread;
    return cone;
}

RayCone propagate(RayCone cone, float distance) { return makeRayCone(abs(cone.width + cone.spread * distance), cone.spread); }

//! A rough lobe widens the cone by about its GGX alpha; a delta lobe keeps it.
RayCone widenRayCone(RayCone cone, float alpha) { return makeRayCone(cone.width, min(cone.spread + 2.0 * alpha, halfPi)); }

uint packRayCone(RayCone cone) { return packHalf2(float2(min(cone.width, 65000.0), cone.spread)); }

RayCone unpackRayCone(uint packed) {
    const float2 values = unpackHalf2(packed);
    return makeRayCone(values.x, values.y);
}

//! Two surface displacements that span the cone footprint on the plane with normal n: the first along the projected ray, stretched by 1/cos.
void coneFootprint(RayCone cone, float3 direction, float3 normal, out float3 dPdx, out float3 dPdy) {
    const float cosine = max(abs(dot(direction, normal)), 0.05);
    float3 along = direction - normal * dot(direction, normal);
    const float alongLength = length(along);
    along = alongLength > 1e-6 ? along / alongLength : makeFrame(normal).tangent;
    const float3 across = cross(normal, along);
    dPdx = along * (cone.width / cosine);
    dPdy = across * cone.width;
}
