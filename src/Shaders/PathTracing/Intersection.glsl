#include "../Bindings.glsl"
#include "../Common.glsl"
#include "ShadeMiss.glsl"
#include "ShadeClosestHit.glsl"

// --- Constants ---
#define EPSILON 1e-5
#define INF 1.0 / 0.0
#define MAX_BVH_STACK_DEPTH 128

// Ray-Primitive Intersection
bool intersectTriangle(vec3 rayOrigin, vec3 rayDirection, vec3 v0, vec3 v1, vec3 v2, inout float t, out vec3 bary) {
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 pvec = cross(rayDirection, e2);
    float det = dot(e1, pvec);

    if (abs(det) < EPSILON)
        return false;
    
    float invDet = 1.0 / det;
    vec3 tvec = rayOrigin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0)
        return false;
    
    vec3 qvec = cross(tvec, e1);
    float v = dot(rayDirection, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0)
        return false;
    
    float current_t = dot(e2, qvec) * invDet;
    if (current_t > EPSILON && current_t < t) {
        t = current_t;
        bary = vec3(1.0 - u - v, u, v);
        return true;
    }

    return false;
}

// Axis-Aligned Bounding Box (AABB) Intersection
bool intersectAABB(vec3 rayOrigin, vec3 invDir, vec3 bbox_min, vec3 bbox_max, float t) {
    vec3 tMin = (bbox_min - rayOrigin) * invDir;
    vec3 tMax = (bbox_max - rayOrigin) * invDir;

    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);

    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);

    return tNear <= tFar && tFar > 0.0f && tNear < t;
}

// BVH Traversal
void traverseBVH(vec3 rayOrigin, vec3 rayDirection,  MeshAddresses mesh, inout HitInfo hit) {
    vec3 invDir = 1.0 / rayDirection;

    // Construct buffer references from the 64-bit addresses within the mesh struct.
    BVHBuffer bvh = BVHBuffer(mesh.blasAddress);
    VertexBuffer vertices = VertexBuffer(mesh.vertexAddress);
    IndexBuffer indices = IndexBuffer(mesh.indexAddress);
    int stack[MAX_BVH_STACK_DEPTH];
    int stackPtr = 0;
    
    stack[stackPtr++] = 0; // Start with the root node.
    while (stackPtr > 0) {
        int nodeIndex = stack[--stackPtr];
        BVHNode node = bvh.data[nodeIndex];
        
        if (!intersectAABB(rayOrigin, invDir, node.bbox.min, node.bbox.max, hit.t))
            continue;
        
        if (node.faceCount > 0) { // Leaf node

            for (int i = 0; i < node.faceCount; ++i) {
              uint primIdx = uint(node.faceIndices[i]);
            
              uint i0 = indices.data[3 * primIdx + 0];
              uint i1 = indices.data[3 * primIdx + 1];
              uint i2 = indices.data[3 * primIdx + 2];
            
              vec3 v0 = vertices.data[i0].position;
              vec3 v1 = vertices.data[i1].position;
              vec3 v2 = vertices.data[i2].position;
            
              vec3 currentBary;
              if (intersectTriangle(rayOrigin, rayDirection, v0, v1, v2, hit.t, currentBary)) {
                  hit.primitiveIndex = int(primIdx);
                  hit.barycentrics = currentBary;
              }
            }
        } else { // Interior node
            if (stackPtr > MAX_BVH_STACK_DEPTH - 2)
                 continue; // Stack is full, cannot traverse deeper.
            
            if (node.leftChild >= 0)
                stack[stackPtr++] = node.leftChild;
            
            if (node.rightChild >= 0)
                stack[stackPtr++] = node.rightChild;
        }
    }
}

HitInfo traceScene(vec3 rayOrigin, vec3 rayDirection) {
    HitInfo bestHit;
    bestHit.t = INF;
    bestHit.instanceIndex = -1;
    bestHit.primitiveIndex = -1;

    for (int i = 0; i < instances.length(); ++i) {
        ComputeInstance inst = instances[i];
        if (inst.meshId == 0xFFFFFFFF)
            continue;

        // Transform ray into local space
        vec3 localOrigin = (inst.inverseTransform * vec4(rayOrigin, 1.0)).xyz;
        vec3 localDir    = normalize((inst.inverseTransform * vec4(rayDirection, 0.0)).xyz);

        HitInfo localHit;
        localHit.t = bestHit.t;   // limit traversal to current best
        localHit.primitiveIndex = -1;

        MeshAddresses mesh = meshes[inst.meshId];
        traverseBVH(localOrigin, localDir, mesh, localHit);

        if (localHit.primitiveIndex != -1) {
            // Convert hit point back to world space to get correct distance
            vec3 localPos  = localOrigin + localDir * localHit.t;
            vec3 worldPos  = (inst.transform * vec4(localPos, 1.0)).xyz;
            float worldT   = length(worldPos - rayOrigin);

            if (worldT < bestHit.t) {
                bestHit.t = worldT;
                bestHit.barycentrics = localHit.barycentrics;
                bestHit.primitiveIndex = localHit.primitiveIndex;
                bestHit.instanceIndex = i;
            }
        }
    }
    return bestHit;
}

void traceRayCompute(vec3 rayOrigin, vec3 rayDirection, inout Payload payload) {
    HitInfo hit = traceScene(rayOrigin, rayDirection);

    if (hit.instanceIndex == -1)
        shadeMiss(rayDirection, pushConstants.environment, payload);
    else {
        const ComputeInstance inst = instances[hit.instanceIndex];
        const MeshAddresses mesh = meshes[inst.meshId];
        const Face face = FaceBuffer(mesh.faceAddress).data[hit.primitiveIndex];
        const Material material = MaterialBuffer(mesh.materialAddress).data[face.materialIndex];
        const Vertex v0 = VertexBuffer(mesh.vertexAddress).data[IndexBuffer(mesh.indexAddress).data[3 * hit.primitiveIndex + 0]];
        const Vertex v1 = VertexBuffer(mesh.vertexAddress).data[IndexBuffer(mesh.indexAddress).data[3 * hit.primitiveIndex + 1]];
        const Vertex v2 = VertexBuffer(mesh.vertexAddress).data[IndexBuffer(mesh.indexAddress).data[3 * hit.primitiveIndex + 2]];
        vec3 localPos = interpolateBarycentric(hit.barycentrics, v0.position, v1.position, v2.position);
        vec3 localNrm = normalize(interpolateBarycentric(hit.barycentrics, v0.normal, v1.normal, v2.normal));
        vec3 localTan = normalize(interpolateBarycentric(hit.barycentrics, v0.tangent, v1.tangent, v2.tangent));
        vec2 uv = interpolateBarycentric(hit.barycentrics, v0.uv, v1.uv, v2.uv);
        vec3 worldPos = (inst.transform * vec4(localPos, 1.0)).xyz;
        mat3 normalMatrix = transpose(inverse(mat3(inst.transform)));
        vec3 interpolatedNormal = normalize(normalMatrix * localNrm);
        vec3 worldTan = normalize(mat3(inst.transform) * localTan);
        shadeClosestHit(worldPos, interpolatedNormal, worldTan, uv, rayDirection, material, payload);
        payload.objectIndex = hit.instanceIndex;
    }
}
