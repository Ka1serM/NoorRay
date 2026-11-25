import NoorRHI;
import SceneRecords;
import Common.Math;
import Common.Frame;
import Common.Random;
import Scene.SceneSurface;

//! The hit whose material the graph evaluates, for the helpers that trace rays; a kernel without a scene leaves the instance invalid, and they trace nothing.
static Ptr<SceneRecord> graphTraceScene;
static uint graphTraceInstance = invalidIndex;
static uint graphTracePrimitive = 0;
static uint graphTraceSeed = 0;

void setGraphTrace(Ptr<SceneRecord> scene, uint instance, uint primitive, uint seed) {
    graphTraceScene = scene;
    graphTraceInstance = instance;
    graphTracePrimitive = primitive;
    graphTraceSeed = seed;
}

bool graphTraces() {
    return graphTraceInstance != invalidIndex;
}

//! Skips the triangle of the hit, as Cycles' ray.self, and with local set every other instance.
struct GraphTraceFilter : IHitFilter {
    uint instance;
    uint primitive;
    bool local;
    bool skipsSelf;
    bool ends;

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        if (skipsSelf && candidate.instanceId == instance && candidate.primitiveIndex == primitive)
            return HitVerdict.Ignore;
        if (local && candidate.instanceId != instance)
            return HitVerdict.Ignore;
        return ends ? HitVerdict.AcceptAndEnd : HitVerdict.Accept;
    }
};

GraphTraceFilter makeGraphTraceFilter(bool local, bool skipsSelf, bool ends) {
    GraphTraceFilter filter;
    filter.instance = graphTraceInstance;
    filter.primitive = graphTracePrimitive;
    filter.local = local;
    filter.skipsSelf = skipsSelf;
    filter.ends = ends;
    return filter;
}

RayDesc makeGraphRay(float3 origin, float3 direction, float tMin, float tMax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    return ray;
}

//! A triangle hit as Cycles' shader setup sees it before it faces the ray: the position, the geometric and the interpolated normal.
struct GraphTraceHit {
    float3 position;
    float3 geometricNormal;
    float3 normal;
};

GraphTraceHit loadGraphTraceHit(SceneRecord scene, uint instanceIndex, uint primitive, float2 barycentrics) {
    const InstanceData instance = scene.instances[instanceIndex];
    const MeshRecord mesh = scene.meshes[instance.meta.x];
    const uint3 vertices = loadTriangle(mesh, primitive);
    const float3 weights = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    const float3 p0 = transformPoint(instance.objectToWorld, loadPosition(mesh, vertices.x));
    const float3 p1 = transformPoint(instance.objectToWorld, loadPosition(mesh, vertices.y));
    const float3 p2 = transformPoint(instance.objectToWorld, loadPosition(mesh, vertices.z));
    const float mirror = determinant3(instance.objectToWorld) < 0.0 ? -1.0 : 1.0;
    GraphTraceHit hit;
    hit.position = interpolate(p0, p1, p2, weights);
    hit.geometricNormal = safeNormalize(cross(p1 - p0, p2 - p0) * mirror);
    hit.normal = hit.geometricNormal;
    if (isNull(mesh.normals))
        return hit;
    const uint3 elements = streamElements(mesh, meshCornerNormals, primitive, vertices);
    const float3 normal = interpolate(unpackOctahedral(mesh.normals[elements.x]), unpackOctahedral(mesh.normals[elements.y]),
                                      unpackOctahedral(mesh.normals[elements.z]), weights);
    const float3 world = safeNormalize(transformNormal(instance.worldToObject, normal));
    if (dot(world, world) > 0.0)
        hit.normal = world;
    return hit;
}

//! The attribute of the object or the instancer that a slot names, 0 for a scene without attributes.
[noinline]
float4 graphInstanceAttribute(float slot) {
    if (!graphTraces())
        return float4(0.0, 0.0, 0.0, 0.0);
    const SceneRecord scene = graphTraceScene[0];
    if (isNull(scene.instanceAttributes))
        return float4(0.0, 0.0, 0.0, 0.0);
    return scene.instanceAttributes[scene.instances[graphTraceInstance].info.w + uint(slot)];
}

//! The vertices of the triangle of the hit, relative to its position.
bool graphTriangle(float3 position, out float3 corners[3]) {
    for (uint corner = 0; corner < 3; ++corner)
        corners[corner] = float3(0.0, 0.0, 0.0);
    if (!graphTraces())
        return false;
    const SceneRecord scene = graphTraceScene[0];
    const InstanceData instance = scene.instances[graphTraceInstance];
    const MeshRecord mesh = scene.meshes[instance.meta.x];
    const uint3 vertices = loadTriangle(mesh, graphTracePrimitive);
    for (uint corner = 0; corner < 3; ++corner)
        corners[corner] = transformPoint(instance.objectToWorld, loadPosition(mesh, vertices[corner])) - position;
    return true;
}

//! The barycentric weights of the hit, which Cycles' Parametric shows.
[noinline]
float3 graphParametric(float3 position) {
    float3 corners[3];
    if (!graphTriangle(position, corners))
        return float3(0.0, 0.0, 0.0);
    const float3 normal = cross(corners[1] - corners[0], corners[2] - corners[0]);
    const float scale = 1.0 / max(dot(normal, normal), 1e-30);
    const float first = dot(cross(corners[1], corners[2]), normal) * scale;
    const float second = dot(cross(corners[2], corners[0]), normal) * scale;
    return float3(first, second, 1.0 - first - second);
}

//! Cycles' wireframe: 1 within half of size of an edge of the triangle.
[noinline]
float graphWireframe(float3 position, float size) {
    float3 corners[3];
    if (!graphTriangle(position, corners))
        return 0.0;
    for (uint corner = 0; corner < 3; ++corner)
        if (length(cross(safeNormalize(corners[corner] - corners[(corner + 2) % 3]), corners[corner])) < size * 0.5)
            return 1.0;
    return 0.0;
}

//! Cycles' svm_ao: settings are the distance, the samples, inside and only local.
[noinline]
float graphAo(float3 position, float3 normal, float4 settings) {
    const uint samples = uint(max(settings.y, 0.0));
    if (!graphTraces() || settings.x <= 0.0 || samples == 0)
        return 1.0;
    const SceneRecord scene = graphTraceScene[0];
    const Frame frame = makeFrame(safeNormalize(normal) * (settings.z > 0.5 ? -1.0 : 1.0));
    Random random = makeRandom(hashCombine(graphTraceSeed, 0x0a0a0a0au));
    uint unoccluded = 0;
    for (uint sample = 0; sample < samples; ++sample) {
        const float radius = sqrt(random.next());
        const float angle = 2.0 * pi * random.next();
        const float2 disk = radius * float2(cos(angle), sin(angle));
        const float3 direction = frame.toWorld(float3(disk, sqrt(max(1.0 - dot(disk, disk), 0.0))));
        GraphTraceFilter filter = makeGraphTraceFilter(settings.w > 0.5, true, true);
        RayHit hit;
        const uint flags = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
        if (!traceRayInline(scene.tlas, flags, 0xffu, makeGraphRay(position, direction, 0.0, settings.x), filter, hit))
            ++unoccluded;
    }
    return float(unoccluded) / float(samples);
}

static const uint localHitLimit = 4u;

//! Keeps up to four hits of the instance along the ray, as Cycles' scene_intersect_local with its reservoir.
struct LocalHitFilter : IHitFilter {
    uint instance;
    uint count;
    Random random;
    float2 barycentrics[4];
    uint primitives[4];

    [mutating]
    HitVerdict filter(RayCandidate candidate) {
        if (candidate.instanceId != instance)
            return HitVerdict.Ignore;
        ++count;
        uint slot = count - 1u;
        if (slot >= localHitLimit) {
            slot = uint(random.next() * float(count));
            if (slot >= localHitLimit)
                return HitVerdict.Ignore;
        }
        barycentrics[slot] = candidate.barycentrics;
        primitives[slot] = candidate.primitiveIndex;
        return HitVerdict.Ignore;
    }
};

float bevelCubic(float radius, float r) {
    if (r >= radius)
        return 0.0;
    const float f = radius - r;
    return 10.0 * f * f * f / (radius * radius * radius * radius * radius * pi);
}

float bevelQuinticRoot(float xi) {
    float x = 0.25;
    for (uint iteration = 0; iteration < 10; ++iteration) {
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float nx = 1.0 - x;
        const float f = 10.0 * x2 - 20.0 * x3 + 15.0 * x2 * x2 - 4.0 * x2 * x3 - xi;
        const float slope = 20.0 * (x * nx) * (nx * nx);
        if (abs(f) < 1e-6 || slope == 0.0)
            break;
        x = saturate(x - f / slope);
    }
    return x;
}

//! Cycles' svm_bevel: settings are the radius, the samples, 1 on a back face, and 1 to skip it on blurry indirect rays; normal is the normal without bevel.
[noinline]
float3 graphBevel(float3 position, float3 geometricNormal, float3 normal, float4 settings) {
    const uint samples = uint(max(settings.y, 0.0));
    const float radius = settings.x;
    if (!graphTraces() || radius <= 0.0 || samples == 0 || settings.w > 0.5)
        return normal;
    const SceneRecord scene = graphTraceScene[0];
    Random random = makeRandom(hashCombine(graphTraceSeed, 0x64c6a40eu));
    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint sample = 0; sample < samples; ++sample) {
        float2 u = float2(random.next(), random.next());
        const Frame base = makeFrame(geometricNormal);
        float3 diskNormal = base.normal;
        float3 diskTangent = base.tangent;
        float3 diskBitangent = base.bitangent;
        float3 pick = float3(0.5, 0.25, 0.25);
        if (u.x < 0.5) {
            u.x *= 2.0;
        } else if (u.x < 0.75) {
            diskNormal = base.tangent;
            diskTangent = base.normal;
            pick = float3(0.25, 0.5, 0.25);
            u.x = (u.x - 0.5) * 4.0;
        } else {
            diskNormal = base.bitangent;
            diskBitangent = base.normal;
            pick = float3(0.25, 0.25, 0.5);
            u.x = (u.x - 0.75) * 4.0;
        }
        const float angle = 2.0 * pi * u.x;
        const float diskRadius = bevelQuinticRoot(u.y) * radius;
        const float height = sqrt(max(radius * radius - diskRadius * diskRadius, 0.0));
        const float3 offset = diskRadius * (cos(angle) * diskTangent + sin(angle) * diskBitangent);
        LocalHitFilter filter;
        filter.instance = graphTraceInstance;
        filter.count = 0;
        filter.random = makeRandom(hashCombine(graphTraceSeed, sample));
        RayHit ignored;
        traceRayInline(scene.tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xffu, makeGraphRay(position + diskNormal * height + offset, -diskNormal, 0.0, 2.0 * height),
                       filter, ignored);
        const uint hits = min(filter.count, localHitLimit);
        for (uint index = 0; index < hits; ++index) {
            const GraphTraceHit hit = loadGraphTraceHit(scene, graphTraceInstance, filter.primitives[index], filter.barycentrics[index]);
            const float pdfN = pick.x * abs(dot(diskNormal, hit.geometricNormal));
            const float pdfT = pick.y * abs(dot(diskTangent, hit.geometricNormal));
            const float pdfB = pick.z * abs(dot(diskBitangent, hit.geometricNormal));
            float weight = pdfN / (pdfN * pdfN + pdfT * pdfT + pdfB * pdfB);
            if (filter.count > localHitLimit)
                weight *= float(filter.count) / float(localHitLimit);
            weight *= bevelCubic(radius, length(hit.position - position)) / bevelCubic(radius, diskRadius);
            sum += weight * hit.normal;
        }
    }
    const float3 result = safeNormalize(sum);
    if (dot(result, result) == 0.0)
        return normal;
    return settings.z > 0.5 ? -result : result;
}

//! Cycles' svm_raycast: settings are the distance, only local, and 1 when the ray starts at the hit, which then skips its triangle.
GraphTraceHit traceGraphRaycast(float3 position, float3 direction, float4 settings, out RayHit hit) {
    hit = (RayHit)0;
    GraphTraceHit result = (GraphTraceHit)0;
    if (!graphTraces() || settings.x <= 0.0)
        return result;
    const SceneRecord scene = graphTraceScene[0];
    GraphTraceFilter filter = makeGraphTraceFilter(settings.y > 0.5, settings.z > 0.5, false);
    if (!traceRayInline(scene.tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xffu, makeGraphRay(position, direction, 0.0, settings.x), filter, hit)) {
        hit.instanceId = invalidIndex;
        return result;
    }
    result = loadGraphTraceHit(scene, hit.instanceId, hit.primitiveIndex, hit.barycentrics);
    if (dot(result.geometricNormal, direction) > 0.0)
        result.normal = -result.normal;
    return result;
}

//! Is Hit, Is Self Hit and Hit Distance of the Raycast node.
[noinline]
float3 graphRaycast(float3 position, float3 direction, float4 settings) {
    RayHit hit;
    traceGraphRaycast(position, direction, settings, hit);
    if (hit.instanceId == invalidIndex || !graphTraces() || settings.x <= 0.0)
        return float3(0.0, 0.0, settings.x);
    return float3(1.0, hit.instanceId == graphTraceInstance ? 1.0 : 0.0, hit.t);
}

//! Hit Normal of the Raycast node, which faces the ray.
[noinline]
float3 graphRaycastNormal(float3 position, float3 direction, float4 settings) {
    RayHit hit;
    const GraphTraceHit result = traceGraphRaycast(position, direction, settings, hit);
    return result.normal;
}
