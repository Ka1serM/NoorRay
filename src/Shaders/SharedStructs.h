#ifdef __cplusplus //In C++, use glm::
    #pragma once
    #include "glm/vec2.hpp"
    #include <glm/vec3.hpp>
    typedef glm::vec2 vec2;
    typedef glm::vec3 vec3;
    typedef glm::uint uint;
#endif


#define BVH_MAX_LEAF_SIZE 4

struct AABB {
    vec3 min;
    vec3 max;

#ifdef __cplusplus
    AABB();
    void expand(const glm::vec3& point);
    void expand(const AABB& other);
    float surfaceArea() const;
    bool intersect(const glm::vec3& origin, const glm::vec3& direction, float tMin, float tMax) const;
#endif
};

struct BVHNode {
#ifdef __cplusplus
    AABB bbox;
#else
    vec3 bbox_min;
    vec3 bbox_max;
#endif
    int leftChild;
    int rightChild;
    int faceCount;
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

#ifdef __cplusplus //C++ only Struct
struct PushConstants {
    PushData push;
    CameraData camera;
};
#endif

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
    vec3 transmission; int _pad1;
    vec3 emission; int _pad2;
    int specularIndex, metallicIndex, roughnessIndex, normalIndex;

#ifdef __cplusplus
    Material()
    : albedo{0.5f}, albedoIndex(-1),
      specular(0.5f), metallic(0), roughness(0.8f), ior(1.5f),
      transmission{0}, _pad1(0),
      emission{0}, _pad2(0),
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
    vec3 color;
    vec3 throughput;
    vec3 position;
    vec3 normal;
    uint hitIndex;
    vec3 nextDirection;
    uint rngState;
    uint bounceType;
    bool done;
};

struct HitInfo {
    float t;
    int instanceIndex;
    int primitiveIndex;
    vec3 barycentrics;
};