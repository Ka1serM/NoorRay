#ifdef __cplusplus
    #pragma once
    #include <glm/vec3.hpp>
    typedef glm::vec3 vec3; //In C++, use glm::vec3 as vec3
#endif

struct PushData {
    int frame, isPathtracing, _pad0, _pad1;
};

struct CameraData {
    vec3 position; int isMoving;
    vec3 direction; int _pad1;
    vec3 horizontal; int _pad2;
    vec3 vertical; int _pad3;
};

struct PointLight {
    vec3 position; float radius;
    vec3 color; float intensity;
};

struct Vertex {
    vec3 position; int _pad0;
    vec3 normal; int _pad1;
};

struct Face {
    int materialIndex, _pad0, _pad1, _pad2;
};

struct Material {
    vec3 albedo; int _pad0;
    float specular, metallic, roughness, ior;
    vec3 transmission; int _pad1;
    vec3 emission; int _pad2;
};

//Vulkan Only

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