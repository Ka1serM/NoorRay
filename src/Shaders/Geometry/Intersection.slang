// Intersection utilities for the shared wavefront raytracers.
// The default path uses software BVH traversal; USE_RTX enables ray-query backed tracing.
#ifndef INTERSECTION_SLANG
#define INTERSECTION_SLANG

#include "../Shared.h"
#include "../Bindings.slang"
#include "../Core/Core.slang"
#include "../Lights/EnvironmentLight.slang"

#ifndef PRIMARY_MISS_INDEX
    #define PRIMARY_MISS_INDEX 0
#endif
#ifndef SHADOW_MISS_INDEX
    #define SHADOW_MISS_INDEX 1
#endif

struct HitInfo
{
    float t;
    uint instanceIndex;
    uint primitiveIndex;
    float3 barycentrics;
};

#ifndef USE_RTX
    #define BVH_STACK_DEPTH 2048

    bool intersectTriangle(float3 rayOrigin, float3 rayDirection, float3 v0, float3 v1, float3 v2, inout float t, out float3 bary, float tMin) {
        const float EPSILON = 1e-8;
        bary = float3(0.0);

        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;

        float3 pvec = cross(rayDirection, e2);
        float det = dot(e1, pvec);

        if (abs(det) < EPSILON)
            return false;

        float invDet = 1.0 / det;
        float3 tvec = rayOrigin - v0;

        float u = dot(tvec, pvec) * invDet;
        float3 qvec = cross(tvec, e1);
        float v = dot(rayDirection, qvec) * invDet;

        if (u < 0.0 || v < 0.0 || u + v > 1.0)
            return false;

        float current_t = dot(e2, qvec) * invDet;

        if (current_t > tMin && current_t < t) {
            t = current_t;
            bary = float3(1.0 - u - v, u, v);
            return true;
        }

        return false;
    }

    bool intersectAABB(float3 rayOrigin, float3 invDir, AABB box, out float tmin, out float tmax) {
        const float EPSILON = 1e-6;

        float3 t0s = (box.minBounds - rayOrigin) * invDir;
        float3 t1s = (box.maxBounds - rayOrigin) * invDir;

        float3 tsmaller = min(t0s, t1s);
        float3 tbigger  = max(t0s, t1s);

        tmin = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
        tmax = min(tbigger.x, min(tbigger.y, tbigger.z));

        return tmin <= tmax + EPSILON && tmax >= 0.0;
    }

    void traverseBVH(float3 rayOrigin, float3 rayDirection, MeshAddresses mesh, inout HitInfo hit, float tMin) {
        float3 invDir = 1.0 / rayDirection;

        BVHNodeBuffer* bvh = reinterpret<BVHNodeBuffer*>(mesh.bvhNodeAddress);
        BVHIndexBuffer* bvhIndices = reinterpret<BVHIndexBuffer*>(mesh.bvhIndexAddress);
        VertexBuffer* vertices = reinterpret<VertexBuffer*>(mesh.vertexAddress);
        IndexBuffer* indices = reinterpret<IndexBuffer*>(mesh.indexAddress);

        uint stack[BVH_STACK_DEPTH];
        uint stackPtr = 0;
        stack[stackPtr++] = 0;

        while (stackPtr > 0) {
            uint nodeIndex = stack[--stackPtr];
            BVHNode node = bvh->data[nodeIndex];

            if (node.primCount > 0) {
                for (uint i = 0; i < node.primCount; ++i) {
                    uint primIdx = bvhIndices->data[node.rightChildOrPrimIndex + i];

                    uint i0 = indices->data[3 * primIdx + 0];
                    uint i1 = indices->data[3 * primIdx + 1];
                    uint i2 = indices->data[3 * primIdx + 2];

                    float3 v0 = vertices->data[i0].position;
                    float3 v1 = vertices->data[i1].position;
                    float3 v2 = vertices->data[i2].position;

                    float3 currentBary;
                    if (intersectTriangle(rayOrigin, rayDirection, v0, v1, v2, hit.t, currentBary, tMin)) {
                        hit.primitiveIndex = int(primIdx);
                        hit.barycentrics = currentBary;
                    }
                }
                continue;
            }

            float tmin_left, tmax_left, tmin_right, tmax_right;
            bool hit_left = intersectAABB(rayOrigin, invDir, node.leftBounds, tmin_left, tmax_left);
            bool hit_right = intersectAABB(rayOrigin, invDir, node.rightBounds, tmin_right, tmax_right);

            hit_left = hit_left && tmin_left < hit.t && tmax_left > tMin;
            hit_right = hit_right && tmin_right < hit.t && tmax_right > tMin;

            if (!hit_left && !hit_right) continue;

            uint leftChildIndex = nodeIndex + 1;
            uint rightChildIndex = node.rightChildOrPrimIndex;

            if (hit_left && hit_right) {
                if (tmin_left < tmin_right) {
                    stack[stackPtr++] = rightChildIndex;
                    stack[stackPtr++] = leftChildIndex;
                } else {
                    stack[stackPtr++] = leftChildIndex;
                    stack[stackPtr++] = rightChildIndex;
                }
            } else if (hit_left) {
                stack[stackPtr++] = leftChildIndex;
            } else if (hit_right) {
                stack[stackPtr++] = rightChildIndex;
            }
        }
    }

    HitInfo traceScene(float3 rayOrigin, float3 rayDirection, float tMin, float tMax) {
        HitInfo bestHit;
        bestHit.t = tMax;
        bestHit.instanceIndex = InvalidInstance;
        bestHit.primitiveIndex = InvalidInstance;

        uint instanceCount = 0;
        uint instanceStride = 0;
        instances.GetDimensions(instanceCount, instanceStride);
        for (uint i = 0; i < instanceCount; ++i) {
            Instance inst = instances[i];

            if (inst.meshId == 0xFFFFFFFF)
                continue;

            float3 localOrigin = mul(inst.inverseTransform, float4(rayOrigin, 1.0)).xyz;
            float3 localDir = mul(inst.inverseTransform, float4(rayDirection, 0.0)).xyz;

            HitInfo localHit;
            localHit.t = bestHit.t;
            localHit.primitiveIndex = InvalidInstance;

            MeshAddresses mesh = meshes[inst.meshId];
            traverseBVH(localOrigin, localDir, mesh, localHit, tMin);

            if (localHit.primitiveIndex != InvalidInstance) {
                float3 localPos = localOrigin + localDir * localHit.t;
                float3 worldPos = mul(inst.transform, float4(localPos, 1.0)).xyz;
                float worldT = length(worldPos - rayOrigin);

                if (worldT >= tMin && worldT < bestHit.t) {
                    bestHit.t = worldT;
                    bestHit.barycentrics = localHit.barycentrics;
                    bestHit.primitiveIndex = localHit.primitiveIndex;
                    bestHit.instanceIndex = i;
                }
            }
        }
        return bestHit;
    }
#endif // !USE_RTX

// Common shadow ray function - works for both Compute and RTX
bool traceShadowRay(float3 rayOrigin, float3 rayDirection, float tMin, float tMax) {
#ifdef USE_RAY_QUERY
    RayDesc shadowRay;
    shadowRay.Origin = rayOrigin;
    shadowRay.Direction = rayDirection;
    shadowRay.TMin = tMin;
    shadowRay.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xFF, shadowRay);
    while (rayQuery.Proceed()) { }
    return rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
#else
    HitInfo shadowHit = traceScene(rayOrigin, rayDirection, tMin, tMax);
    return shadowHit.instanceIndex != InvalidInstance;
#endif
}

// Common reflection ray function - works for both Compute and RTX
bool traceReflectionRay(float3 rayOrigin, float3 rayDirection, float roughness, EnvironmentSettings environmentData, out float3 hitColor) {
#ifndef USE_RTX
    HitInfo hit = traceScene(rayOrigin, rayDirection, 0.001, 1000.0);

    if (hit.instanceIndex == InvalidInstance) {
        hitColor = sampleEnvironmentMapColorLod(rayDirection, environmentData, environmentData.lightingExposureScale, roughness * max(environmentData.maxTextureLod, 0.0));
        return false;
    }

    const Instance inst = instances[hit.instanceIndex];
    const MeshAddresses mesh = meshes[inst.meshId];
    const Face face = reinterpret<FaceBuffer*>(mesh.faceAddress)->data[hit.primitiveIndex];
    const Material material = reinterpret<MaterialBuffer*>(mesh.materialAddress)->data[face.materialIndex];

    const Vertex v0 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * hit.primitiveIndex + 0]];
    const Vertex v1 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * hit.primitiveIndex + 1]];
    const Vertex v2 = reinterpret<VertexBuffer*>(mesh.vertexAddress)->data[reinterpret<IndexBuffer*>(mesh.indexAddress)->data[3 * hit.primitiveIndex + 2]];

    float2 uv = interpolateBarycentric(hit.barycentrics, v0.uv, v1.uv, v2.uv);

    float3 albedo = material.albedo;
    if (material.albedoIndex != -1)
        albedo *= textureSamplers[material.albedoIndex].Sample(uv).rgb;

    float3 emission = material.emission * material.emissionStrength;
    if (material.emissionIndex != -1)
        emission *= textureSamplers[material.emissionIndex].Sample(uv).rgb;

    hitColor = emission + albedo * 0.5;
    return true;
#else
    // RTX path falls back to the environment sample here; ray-query shaders do not use a separate payload path.
    hitColor = sampleEnvironmentMapColorLod(rayDirection, environmentData, environmentData.lightingExposureScale, roughness * max(environmentData.maxTextureLod, 0.0));
    return false;
#endif
}

float3 computeShadowTerminatorPointLocal(
    float3 localPos,
    float3 bary,
    float3 v0Pos, float3 v1Pos, float3 v2Pos,
    float3 v0Normal, float3 v1Normal, float3 v2Normal)
{
    float3 n0 = normalize(v0Normal);
    float3 n1 = normalize(v1Normal);
    float3 n2 = normalize(v2Normal);

    float3 tmp0 = localPos - v0Pos;
    float3 tmp1 = localPos - v1Pos;
    float3 tmp2 = localPos - v2Pos;

    float d0 = min(0.0, dot(tmp0, n0));
    float d1 = min(0.0, dot(tmp1, n1));
    float d2 = min(0.0, dot(tmp2, n2));

    tmp0 -= d0 * n0;
    tmp1 -= d1 * n1;
    tmp2 -= d2 * n2;

    float3 offset = bary.x * tmp0 + bary.y * tmp1 + bary.z * tmp2;
    float maxOffset = 0.2 * max(
        length(v1Pos - v0Pos),
        max(length(v2Pos - v1Pos), length(v0Pos - v2Pos)));
    float offsetLength = length(offset);
    if (offsetLength > maxOffset)
        offset *= maxOffset / max(offsetLength, EPSILON);

    return localPos + offset;
}

#endif // INTERSECTION_SLANG
