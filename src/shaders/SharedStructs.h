#ifdef __cplusplus //In C++, use glm::
    #pragma once
    #include "glm/vec2.hpp"
    #include <glm/vec3.hpp>
    typedef glm::vec2 vec2;
    typedef glm::vec3 vec3;
#endif

struct PushData {
    int frame, isRayTracing, _pad0, _pad1;
};

struct CameraData {
    vec3 position; int isMoving;
    vec3 direction; float aperture;
    vec3 horizontal; float focusDistance;
    vec3 vertical; float focalLength;
};

#ifdef __cplusplus //C++ only Struct
struct PushConstants {
    PushData push;
    CameraData camera;
};
#endif

struct PointLight {
    vec3 position; float radius;
    vec3 color; float intensity;
};

struct Vertex {
    vec3 position; int _pad0;
    vec3 normal; int _pad1;
    vec2 uv; int _pad2, _pad3;
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
};

//Vulkan Only

struct MeshAddresses {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint64_t faceAddress;
};

struct PrimaryRayPayload
{
    vec3 color;
    vec3 throughput;
    vec3 position;
    vec3 normal;
    bool done;
};

struct ShadowRayPayload
{
    bool hit;
};