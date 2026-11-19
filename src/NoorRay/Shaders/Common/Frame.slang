import NoorRHI;

//! An orthonormal frame with the normal as +Z.
struct Frame {
    float3 tangent;
    float3 bitangent;
    float3 normal;

    float3 toLocal(float3 v) { return float3(dot(v, tangent), dot(v, bitangent), dot(v, normal)); }
    float3 toWorld(float3 v) { return tangent * v.x + bitangent * v.y + normal * v.z; }
};

//! Duff et al. 2017, "Building an Orthonormal Basis, Revisited".
Frame makeFrame(float3 normal) {
    const float s = normal.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + normal.z);
    const float b = normal.x * normal.y * a;
    Frame frame;
    frame.tangent = float3(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x);
    frame.bitangent = float3(b, s + normal.y * normal.y * a, -normal.y);
    frame.normal = normal;
    return frame;
}

//! Gram-Schmidt of the tangent against the normal; falls back to makeFrame when the tangent is parallel.
Frame makeFrame(float3 normal, float3 tangent) {
    const float3 projected = tangent - normal * dot(normal, tangent);
    const float lengthSquared = dot(projected, projected);
    if (lengthSquared < 1e-12)
        return makeFrame(normal);
    Frame frame;
    frame.normal = normal;
    frame.tangent = projected * rsqrt(lengthSquared);
    frame.bitangent = cross(normal, frame.tangent);
    return frame;
}
