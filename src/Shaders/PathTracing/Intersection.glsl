#include "../Common.glsl" 

// Intersects a ray with a single triangle using the Möller-Trumbore algorithm.
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

// Intersects a ray with an Axis-Aligned Bounding Box.
bool intersectAABB(vec3 rayOrigin, vec3 invDir, vec3 bbox_min, vec3 bbox_max, inout float t) {
    vec3 t1 = (bbox_min - rayOrigin) * invDir;
    vec3 t2 = (bbox_max - rayOrigin) * invDir;

    // Conditionally swap t-values based on ray direction
    if (invDir.x < 0.0f) { float temp = t1.x; t1.x = t2.x; t2.x = temp; }
    if (invDir.y < 0.0f) { float temp = t1.y; t1.y = t2.y; t2.y = temp; }
    if (invDir.z < 0.0f) { float temp = t1.z; t1.z = t2.z; t2.z = temp; }

    // After swapping, t1 contains the near plane intersections and t2 contains the far ones.
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);

    // An intersection occurs if the ray enters the box before it exits,
    // the box is in front of the ray, and the intersection is closer than the current closest hit.
    return tNear <= tFar && tFar > 0.0f && tNear < t;
}

// Traverses a single BVH to find the closest hit triangle for a mesh.
void traverseBVH(vec3 rayOrigin, vec3 rayDirection, BVHBuffer bvh, VertexBuffer vertexBuf, IndexBuffer indexBuf, inout HitInfo hit) {
    vec3 invDir = 1.0 / rayDirection;
    int stack[128];
    int stackPtr = 1;
    stack[0] = 0; // Start with the root node

    while (stackPtr > 0) {
        BVHNode node = bvh.data[stack[--stackPtr]];

        if (!intersectAABB(rayOrigin, invDir, node.bbox.min, node.bbox.max, hit.t))
            continue;

        if (node.faceCount > 0) { // Leaf node
                                  for (int i = 0; i < node.faceCount; ++i) {
                                      uint primIdx = uint(node.faceIndices[i]);
                                      uint i0 = indexBuf.data[3 * primIdx + 0];
                                      uint i1 = indexBuf.data[3 * primIdx + 1];
                                      uint i2 = indexBuf.data[3 * primIdx + 2];

                                      vec3 currentBary;
                                      if (intersectTriangle(rayOrigin, rayDirection, vertexBuf.data[i0].position, vertexBuf.data[i1].position, vertexBuf.data[i2].position, hit.t, currentBary)) {
                                          hit.primitiveIndex = int(primIdx);
                                          hit.barycentrics = currentBary;
                                      }
                                  }
        } else { // Branch node
                 stack[stackPtr++] = node.rightChild;
                 stack[stackPtr++] = node.leftChild;
        }
    }
}

// Traces a ray through all instances in the scene.
HitInfo traceScene(vec3 rayOrigin, vec3 rayDirection) {
    HitInfo hit;
    hit.t = INF;
    hit.instanceIndex = -1;
    hit.primitiveIndex = -1;

    for (int i = 0; i < sceneInstances.length(); ++i) {
        ComputeInstance inst = sceneInstances[i];

        if (inst.meshId >= uint(instances.length())) // Invalid instance
            continue;

        vec3 localOrigin = (inst.inverseTransform * vec4(rayOrigin, 1.0)).xyz;
        vec3 localDir = (inst.inverseTransform * vec4(rayDirection, 0.0)).xyz;

        HitInfo localHit;
        localHit.t = hit.t;
        localHit.primitiveIndex = -1;

        MeshAddresses mesh = instances[inst.meshId];
        traverseBVH(localOrigin, normalize(localDir), BVHBuffer(mesh.blasAddress), VertexBuffer(mesh.vertexAddress), IndexBuffer(mesh.indexAddress), localHit);

        if (localHit.primitiveIndex != -1 && localHit.t < hit.t) {
            hit.t = localHit.t;
            hit.barycentrics = localHit.barycentrics;
            hit.primitiveIndex = localHit.primitiveIndex;
            hit.instanceIndex = i;
        }
    }
    return hit;
}