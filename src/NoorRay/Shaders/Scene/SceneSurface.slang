import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Frame;
import Common.RayCone;

//! A mesh hit as the shade kernels see it. context.normal and context.geometricNormal face the viewer; worldNormal keeps the winding of the world-space triangle.
struct Surface {
    ShadingContext context;
    float3 worldNormal;
    //! The hit lifted onto the smooth surface that the vertex normals describe, on the viewer side.
    float3 liftedPosition;
    float3 previousPosition;
    uint instance;
    uint primitive;
    uint material;
    uint materialFlags;
    uint instanceFlags;
    uint light;
    float area;
};

float3 loadPosition(MeshRecord mesh, uint vertex) {
    const Ptr<float> positions = mesh.positions + vertex * 3;
    return float3(positions[0], positions[1], positions[2]);
}

float2 loadUv(Ptr<uint> stream, uint vertex) { return isNull(stream) ? float2(0.0, 0.0) : unpackHalf2(stream[vertex]); }

float2 interpolate(float2 a, float2 b, float2 c, float3 weights) { return a * weights.x + b * weights.y + c * weights.z; }
float3 interpolate(float3 a, float3 b, float3 c, float3 weights) { return a * weights.x + b * weights.y + c * weights.z; }
float4 interpolate(float4 a, float4 b, float4 c, float4 weights) { return a * weights.x + b * weights.y + c * weights.z; }

uint triangleMaterial(InstanceData instance, MeshRecord mesh, uint primitive) {
    return instance.meta.z != invalidIndex ? instance.meta.z : mesh.triangleMaterials[primitive];
}

uint triangleMaterial(SceneRecord scene, uint instanceIndex, uint primitive) {
    const InstanceData instance = scene.instances[instanceIndex];
    return triangleMaterial(instance, scene.meshes[instance.meta.x], primitive);
}

uint3 loadTriangle(MeshRecord mesh, uint primitive) {
    const Ptr<uint> indices = mesh.indices + primitive * 3;
    return uint3(indices[0], indices[1], indices[2]);
}

//! The barycentric change for a world-space displacement d within the triangle plane spanned by e1 and e2.
float2 barycentricDelta(float3 e1, float3 e2, float3 d) {
    const float a = dot(e1, e1);
    const float b = dot(e1, e2);
    const float c = dot(e2, e2);
    const float determinant = a * c - b * b;
    if (abs(determinant) < 1e-20)
        return float2(0.0, 0.0);
    const float p = dot(d, e1);
    const float q = dot(d, e2);
    return float2(c * p - b * q, a * q - b * p) / determinant;
}

float determinant3(float4 rows[3]) { return dot(rows[0].xyz, cross(rows[1].xyz, rows[2].xyz)); }

//! barycentrics are the hardware's (weights of vertices 1 and 2). direction is the normalized world ray, cone already propagated to the hit.
Surface loadSurface(SceneRecord scene, uint instanceIndex, uint primitive, float2 barycentrics, float3 direction, float distance, RayCone cone) {
    const InstanceData instance = scene.instances[instanceIndex];
    const MeshRecord mesh = scene.meshes[instance.meta.x];
    const uint3 vertices = loadTriangle(mesh, primitive);
    const float3 weights = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    const float3 o0 = loadPosition(mesh, vertices.x);
    const float3 o1 = loadPosition(mesh, vertices.y);
    const float3 o2 = loadPosition(mesh, vertices.z);
    const float3 objectPosition = interpolate(o0, o1, o2, weights);
    const float3 p0 = transformPoint(instance.objectToWorld, o0);
    const float3 p1 = transformPoint(instance.objectToWorld, o1);
    const float3 p2 = transformPoint(instance.objectToWorld, o2);
    const float3 e1 = p1 - p0;
    const float3 e2 = p2 - p0;
    const float mirror = determinant3(instance.objectToWorld) < 0.0 ? -1.0 : 1.0;
    const float3 crossProduct = cross(e1, e2) * mirror;
    const float crossLength = length(crossProduct);
    const float3 worldNormal = crossLength > 0.0 ? crossProduct / crossLength : -direction;

    Surface surface;
    surface.worldNormal = worldNormal;
    surface.instance = instanceIndex;
    surface.primitive = primitive;
    surface.instanceFlags = instance.meta.y;
    surface.area = 0.5 * crossLength;
    surface.material = triangleMaterial(instance, mesh, primitive);
    surface.materialFlags = scene.materials[surface.material].flags;
    surface.light = instance.meta.w == invalidIndex ? invalidIndex : instance.meta.w + primitive;
    surface.previousPosition = transformPoint(instance.previousObjectToWorld, objectPosition);

    const float3 position = transformPoint(instance.objectToWorld, objectPosition);
    const bool backfacing = dot(worldNormal, direction) > 0.0;
    const float3 geometricNormal = backfacing ? -worldNormal : worldNormal;

    float3 normal = worldNormal;
    surface.liftedPosition = position;
    if (!isNull(mesh.normals)) {
        const float3 v0 = unpackOctahedral(mesh.normals[vertices.x]);
        const float3 v1 = unpackOctahedral(mesh.normals[vertices.y]);
        const float3 v2 = unpackOctahedral(mesh.normals[vertices.z]);
        normal = safeNormalize(transformNormal(instance.worldToObject, interpolate(v0, v1, v2, weights)));
        if (dot(normal, normal) == 0.0)
            normal = worldNormal;
        //! Hanika 2021: each vertex normal spans a tangent plane, and a hit below a plane moves up to it, weighted like the vertex.
        const float side = backfacing ? -1.0 : 1.0;
        const float3 n0 = side * safeNormalize(transformNormal(instance.worldToObject, v0));
        const float3 n1 = side * safeNormalize(transformNormal(instance.worldToObject, v1));
        const float3 n2 = side * safeNormalize(transformNormal(instance.worldToObject, v2));
        surface.liftedPosition = position - weights.x * min(dot(position - p0, n0), 0.0) * n0 - weights.y * min(dot(position - p1, n1), 0.0) * n1 -
                                 weights.z * min(dot(position - p2, n2), 0.0) * n2;
    }

    float4 tangent;
    if (!isNull(mesh.tangents)) {
        const float4 t0 = unpackTangent(mesh.tangents[vertices.x]);
        const float4 t1 = unpackTangent(mesh.tangents[vertices.y]);
        const float4 t2 = unpackTangent(mesh.tangents[vertices.z]);
        const float3 t = transformVector(instance.objectToWorld, interpolate(t0.xyz, t1.xyz, t2.xyz, weights));
        tangent = float4(safeNormalize(t - normal * dot(normal, t)), t0.w * mirror);
    } else {
        const float2 uv0 = loadUv(mesh.uv0, vertices.x);
        const float2 duv1 = loadUv(mesh.uv0, vertices.y) - uv0;
        const float2 duv2 = loadUv(mesh.uv0, vertices.z) - uv0;
        const float uvDeterminant = duv1.x * duv2.y - duv1.y * duv2.x;
        float3 t = abs(uvDeterminant) > 1e-12 ? (e1 * duv2.y - e2 * duv1.y) / uvDeterminant : makeFrame(normal).tangent;
        t = t - normal * dot(normal, t);
        tangent = float4(dot(t, t) > 1e-20 ? normalize(t) : makeFrame(normal).tangent, uvDeterminant < 0.0 ? -1.0 : 1.0);
    }

    float3 dPdx;
    float3 dPdy;
    coneFootprint(cone, direction, worldNormal, dPdx, dPdy);
    const float2 bx = barycentricDelta(e1, e2, dPdx);
    const float2 by = barycentricDelta(e1, e2, dPdy);

    ShadingContext context;
    context.position = float4(position, 1.0);
    context.normal = float4(backfacing ? -normal : normal, backfacing ? 1.0 : 0.0);
    context.geometricNormal = float4(geometricNormal, 0.0);
    context.tangent = tangent;
    context.incoming = float4(-direction, distance);

    float2 uvs[4];
    uvs[0] = float2(0.0, 0.0);
    uvs[1] = float2(0.0, 0.0);
    uvs[2] = float2(0.0, 0.0);
    uvs[3] = float2(0.0, 0.0);
    float2 uvGradientX = float2(0.0, 0.0);
    float2 uvGradientY = float2(0.0, 0.0);
    [unroll]
    for (uint set = 0; set < 4; ++set) {
        if (set >= mesh.uvSetCount)
            break;
        const Ptr<uint> stream = set == 0 ? mesh.uv0 : (set == 1 ? mesh.uv1 : (set == 2 ? mesh.uv2 : mesh.uv3));
        const float2 a = loadUv(stream, vertices.x);
        const float2 b = loadUv(stream, vertices.y);
        const float2 c = loadUv(stream, vertices.z);
        uvs[set] = interpolate(a, b, c, weights);
        if (set == 0) {
            uvGradientX = (b - a) * bx.x + (c - a) * bx.y;
            uvGradientY = (b - a) * by.x + (c - a) * by.y;
        }
    }
    context.uv01 = float4(uvs[0], uvs[1]);
    context.uv23 = float4(uvs[2], uvs[3]);
    context.uvGradients = float4(uvGradientX, uvGradientY);

    context.color = float4(1.0, 1.0, 1.0, 1.0);
    if (!isNull(mesh.colors))
        context.color = interpolate(unpackUnorm4x8(mesh.colors[vertices.x]), unpackUnorm4x8(mesh.colors[vertices.y]), unpackUnorm4x8(mesh.colors[vertices.z]),
                                    float4(weights, 0.0));

    context.dPdx = float4(dPdx, 0.0);
    context.dPdy = float4(dPdy, 0.0);
    surface.context = context;
    return surface;
}

//! Wächter and Binder 2019: moves p off the surface along n by a few ulps, scaled with the magnitude of p.
float3 offsetRayOrigin(float3 p, float3 n) {
    const float origin = 1.0 / 32.0;
    const float floatScale = 1.0 / 65536.0;
    const float intScale = 256.0;
    const int3 offset = int3(intScale * n);
    const float3 shifted = float3(asfloat(asint(p.x) + (p.x < 0.0 ? -offset.x : offset.x)), asfloat(asint(p.y) + (p.y < 0.0 ? -offset.y : offset.y)),
                                  asfloat(asint(p.z) + (p.z < 0.0 ? -offset.z : offset.z)));
    return float3(abs(p.x) < origin ? p.x + floatScale * n.x : shifted.x, abs(p.y) < origin ? p.y + floatScale * n.y : shifted.y,
                  abs(p.z) < origin ? p.z + floatScale * n.z : shifted.z);
}
