import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Frame;
import Common.RayCone;
import Common.Random;

//! A mesh hit for the shade kernels: context.normal and context.geometricNormal face the viewer; worldNormal keeps the triangle winding.
struct Surface {
    ShadingContext context;
    float3 worldNormal;
    //! The hit lifted onto the smooth surface that the shading normals describe, on the viewer side.
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

//! The stream elements of the three corners of a triangle: its corners for a corner stream, its vertices for a vertex stream.
uint3 streamElements(MeshRecord mesh, uint stream, uint primitive, uint3 vertices) {
    return (mesh.cornerStreams & stream) != 0 ? uint3(primitive * 3, primitive * 3 + 1, primitive * 3 + 2) : vertices;
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
        const uint3 elements = streamElements(mesh, meshCornerNormals, primitive, vertices);
        const float3 v0 = unpackOctahedral(mesh.normals[elements.x]);
        const float3 v1 = unpackOctahedral(mesh.normals[elements.y]);
        const float3 v2 = unpackOctahedral(mesh.normals[elements.z]);
        normal = safeNormalize(transformNormal(instance.worldToObject, interpolate(v0, v1, v2, weights)));
        if (dot(normal, normal) == 0.0)
            normal = worldNormal;
        //! Hanika 2021: each corner normal spans a tangent plane, and a hit below a plane moves up to it, weighted like the vertex.
        const float side = backfacing ? -1.0 : 1.0;
        const float3 n0 = side * safeNormalize(transformNormal(instance.worldToObject, v0));
        const float3 n1 = side * safeNormalize(transformNormal(instance.worldToObject, v1));
        const float3 n2 = side * safeNormalize(transformNormal(instance.worldToObject, v2));
        surface.liftedPosition = position - weights.x * min(dot(position - p0, n0), 0.0) * n0 - weights.y * min(dot(position - p1, n1), 0.0) * n1 -
                                 weights.z * min(dot(position - p2, n2), 0.0) * n2;
    }

    float3 restPosition = objectPosition;
    if (!isNull(mesh.restPositions)) {
        const float3 r0 = loadFloat3(mesh.restPositions, vertices.x);
        const float3 r1 = loadFloat3(mesh.restPositions, vertices.y);
        const float3 r2 = loadFloat3(mesh.restPositions, vertices.z);
        restPosition = interpolate(r0, r1, r2, weights);
        if ((surface.materialFlags & materialFlagRestNormal) != 0) {
            float3 rest = cross(r1 - r0, r2 - r0);
            if (!isNull(mesh.normals))
                rest = interpolate(loadFloat3(mesh.restNormals, vertices.x), loadFloat3(mesh.restNormals, vertices.y), loadFloat3(mesh.restNormals, vertices.z), weights);
            rest = safeNormalize(transformNormal(instance.worldToObject, rest));
            normal = dot(rest, rest) > 0.0 ? rest : normal;
        }
    }

    const uint3 uvElements = streamElements(mesh, meshCornerUvs, primitive, vertices);
    float4 tangent;
    if (!isNull(mesh.tangents)) {
        const uint3 elements = streamElements(mesh, meshCornerTangents, primitive, vertices);
        const float4 t0 = unpackTangent(mesh.tangents[elements.x]);
        const float4 t1 = unpackTangent(mesh.tangents[elements.y]);
        const float4 t2 = unpackTangent(mesh.tangents[elements.z]);
        const float3 t = transformVector(instance.objectToWorld, interpolate(t0.xyz, t1.xyz, t2.xyz, weights));
        tangent = float4(safeNormalize(t - normal * dot(normal, t)), t0.w * mirror);
    } else {
        const float2 uv0 = loadUv(mesh.uv0, uvElements.x);
        const float2 duv1 = loadUv(mesh.uv0, uvElements.y) - uv0;
        const float2 duv2 = loadUv(mesh.uv0, uvElements.z) - uv0;
        const float uvDeterminant = duv1.x * duv2.y - duv1.y * duv2.x;
        float3 t = abs(uvDeterminant) > 1e-12 ? (e1 * duv2.y - e2 * duv1.y) / uvDeterminant : makeFrame(normal).tangent;
        t = t - normal * dot(normal, t);
        //! The sign 0 tells a normal map that the mesh has no UV map.
        tangent = float4(dot(t, t) > 1e-20 ? normalize(t) : makeFrame(normal).tangent, isNull(mesh.uv0) ? 0.0 : uvDeterminant < 0.0 ? -1.0 : 1.0);
    }

    float3 dPdx;
    float3 dPdy;
    coneFootprint(cone, direction, worldNormal, dPdx, dPdy);
    const float2 bx = barycentricDelta(e1, e2, dPdx);
    const float2 by = barycentricDelta(e1, e2, dPdy);
    if ((surface.materialFlags & materialFlagRestNormal) != 0) {
        dPdx -= normal * dot(normal, dPdx);
        dPdy -= normal * dot(normal, dPdy);
    }

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
        const Ptr<uint> stream = set == 0 ? mesh.uv0 : (set == 1 ? mesh.uv1 : (set == 2 ? mesh.uv2 : mesh.uv3));
        if (isNull(stream))
            continue;
        const float2 a = loadUv(stream, uvElements.x);
        const float2 b = loadUv(stream, uvElements.y);
        const float2 c = loadUv(stream, uvElements.z);
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
    if (!isNull(mesh.colors)) {
        const uint3 elements = streamElements(mesh, meshCornerColors, primitive, vertices);
        context.color = interpolate(unpackUnorm4x8(mesh.colors[elements.x]), unpackUnorm4x8(mesh.colors[elements.y]), unpackUnorm4x8(mesh.colors[elements.z]),
                                    float4(weights, 0.0));
    }

    context.dPdx = float4(dPdx, 0.0);
    context.dPdy = float4(dPdy, 0.0);
    //! Blender's automatic texture space gives a flat axis a half size of 1 about its middle.
    const float3 extent = mesh.boundsMax.xyz - mesh.boundsMin.xyz;
    const bool3 flat = extent == 0.0;
    const float3 generatedScale = 1.0 / select(flat, float3(2.0, 2.0, 2.0), extent);
    const float3 generated = (restPosition - select(flat, mesh.boundsMin.xyz - 1.0, mesh.boundsMin.xyz)) * generatedScale;
    const uint closureSeed = hashCombine(hashCombine(hashPcg(asuint(objectPosition.x)), asuint(objectPosition.y)), asuint(objectPosition.z) ^ primitive);
    context.objectPosition = float4((surface.materialFlags & materialFlagRestNormal) != 0 ? restPosition : objectPosition, float(instance.info.z >> 16) / 65535.0);
    context.generated = float4(generated, uintToFloat(closureSeed));
    context.objectLocation = float4(instance.objectToWorld[0].w, instance.objectToWorld[1].w, instance.objectToWorld[2].w, 0.0);
    context.objectDx = float4(transformVector(instance.worldToObject, dPdx), 0.0);
    context.objectDy = float4(transformVector(instance.worldToObject, dPdy), 0.0);
    context.generatedScale = float4(generatedScale, 0.0);
    context.worldToObject0 = instance.worldToObject[0];
    context.worldToObject1 = instance.worldToObject[1];
    context.worldToObject2 = instance.worldToObject[2];
    context.objectToWorld0 = instance.objectToWorld[0];
    context.objectToWorld1 = instance.objectToWorld[1];
    context.objectToWorld2 = instance.objectToWorld[2];
    setCameraRay(context);
    context.objectColor = float4(unpackHalf2(instance.info.x), unpackHalf2(instance.info.y));
    context.objectInfo.x = float(instance.info.z & 0xffffu);
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
