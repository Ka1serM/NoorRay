#ifdef __cplusplus //In C++, use glm::
    #pragma once
    #include "glm/vec2.hpp"
    #include <glm/vec3.hpp>
    #include <glm/mat4x4.hpp>
    using namespace glm;
#endif


#define BVH_MAX_LEAF_SIZE 4

struct AABB {
    vec3 min;
    float _pad0;
    vec3 max;
    float _pad1;
#ifdef __cplusplus
    AABB();
    void expand(const vec3& point);
    void expand(const AABB& other);
    float surfaceArea() const;
    bool intersect(const vec3& origin, const vec3& invDir, const ivec3& dirIsNeg, float& tNear, float& tFar) const;
#endif
};

struct BVHNode {
    AABB bbox;
    int leftChild;
    int rightChild;
    int faceCount;
    int _pad2; // Add padding to align the array
    int faceIndices[BVH_MAX_LEAF_SIZE];

#ifdef __cplusplus
    bool isLeaf() const { return faceCount > 0; }
#endif
};


struct PushData {
    int frame, isMoving, hdriTexture, _pad0;
};

struct CameraData {
    vec3 position; float aperture;
    vec3 direction; float focusDistance;
    vec3 horizontal; float focalLength;
    vec3 vertical; float bokehBias;
};

struct PushConstantsData {
    PushData push;
    CameraData camera;
};

struct Vertex {
    vec3 position; int _pad0;
    vec3 normal; int _pad1;
    vec3 tangent; int _pad2;
    vec2 uv; int _pad3, _pad4;
};

struct Face {
    int materialIndex, _pad0, _pad1, _pad2;
};

struct Material {
    vec3 albedo; int albedoIndex;
    float specular, metallic, roughness, ior;
    int specularIndex, metallicIndex, roughnessIndex, normalIndex;
    vec3 transmission; float transmissionStrength;
    vec3 emission; float emissionStrength;

#ifdef __cplusplus
    Material()
    : albedo{0.5f}, albedoIndex(-1),
      specular(0.5f), metallic(0), roughness(0.8f), ior(1.5f),
      transmission{1}, transmissionStrength(0),
      emission{1}, emissionStrength(0),
      specularIndex(-1), metallicIndex(-1), roughnessIndex(-1), normalIndex(-1)
    {}
#endif
};

//Vulkan Only
struct MeshAddresses {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint64_t faceAddress;
    uint64_t materialAddress;
    uint64_t blasAddress;
};

struct Payload
{
    vec3 albedo; int objectIndex;
    vec3 normal; float roughness;
    vec3 color; uint rngState;
    vec3 throughput; bool done;
    vec3 position; uint bounceType; 
    vec3 nextDirection; int _pad0;
};

struct HitInfo {
    float t;
    int instanceIndex;
    int primitiveIndex;
    vec3 barycentrics;
};

struct ComputeInstance {
    mat4 transform;
    mat4 inverseTransform;
    uint meshId, pad1, pad2, pad3; 
};