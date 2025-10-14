#ifdef __cplusplus //In C++, use 
    #pragma once
    #include "glm/vec2.hpp"
    #include <glm/vec3.hpp>
    #include <glm/mat4x4.hpp>
    using namespace glm;
#else
    #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

#define INVALID_INSTANCE 0xFFFFFFFFu //max uint32_t
#define GROUP_SIZE 16

#define MAX_LEAF_SIZE 8
#define SAH_BINS 16

struct AABB
{
    vec3 minBounds;
    float _pad0;
    vec3 maxBounds;
    float _pad1;

#ifdef __cplusplus
    AABB()
        : minBounds(std::numeric_limits<float>::max()),
          _pad0(0.0f),
          maxBounds(-std::numeric_limits<float>::max()),
          _pad1(0.0f)
    {}

    void expand(const vec3& point) {
        minBounds = min(minBounds, point);
        maxBounds = max(maxBounds, point);
    }

    void expand(const AABB& other) {
        minBounds = min(minBounds, other.minBounds);
        maxBounds = max(maxBounds, other.maxBounds);
    }

    float surfaceArea() const {
        vec3 d = maxBounds - minBounds;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    vec3 centroid() const {
        return (minBounds + maxBounds) * 0.5f;
    }
#endif
};

// Node is 64 bytes, aligning perfectly to GPU cache lines.
struct  BVHNode {
    AABB leftBounds;  // 32 bytes
    AABB rightBounds; // 32 bytes
    // For a leaf node: index of the first primitive in the flat index list.
    uint rightChildOrPrimIndex;
    // For an interior node: 0
    // For a leaf node: number of primitives in the leaf.
    uint primCount;
    // For an interior node: split axis (0=x, 1=y, 2=z)
    // For a leaf node: unused
    uint splitAxis;
    uint _pad[7]; // Padding
};

struct PushData {
    int samples, diffuseBounces, specularBounces, transmissionBounces;
    float exposure; int frame, isMoving, visualizeBVH;
};

struct EnvironmentData {
    vec3 color; float intensity;
    int textureIndex; int cdfTextureIndex; float rotation; float exposure;
    int visible, _pad0, _pad1,  _pad2;
#ifdef __cplusplus
    EnvironmentData()
    : color(1), intensity(1), textureIndex(0), cdfTextureIndex(-1), rotation(0), exposure(0), visible(1), _pad0(0), _pad1(0), _pad2(0)
    {}
#endif
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
    EnvironmentData environment;
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
    vec3 transmissionColor; float transmission;
    vec3 emission; float emissionStrength;
    int emissionIndex, transmissionIndex, opacityIndex; float opacity;

#ifdef __cplusplus
    Material()
        : albedo{1}, albedoIndex(-1),
          specular(0.5f), metallic(0), roughness(0), ior(1.5f),
          specularIndex(-1), metallicIndex(-1),
          roughnessIndex(-1), normalIndex(-1),
          transmissionColor(1), transmission(0), emission(1),
          emissionStrength(0), emissionIndex(-1), transmissionIndex(-1), opacityIndex(-1), opacity(1)
    {}
#endif
};

struct AssetData {
    uint type;
    uint64_t bufferDeviceAddress;
};

//Vulkan Only
struct MeshData {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint64_t faceAddress;
    uint64_t materialAddress;
    // Compute RT
    uint64_t bvhNodeAddress;
    uint64_t bvhIndexAddress;
};

struct VolumeData {
    uint64_t voxelBufferAddress;   // Device address of 3D image / SSBO
};

// Sampler struct to keep track of state for the current pixel.
struct SamplerState {
    vec2 randomOffset;
    int frame;
    int dimension;
};

struct Payload {
    vec3 attenuation; uint flags;
    vec3 emission; int pad0;
    
    vec3 position; uint depth; 
    vec3 nextDirection; uint rngState;
    
    vec3 albedo; float roughness;
    vec3 normal; uint objectIndex;
};


struct HitInfo {
    float t;
    uint instanceIndex;
    uint primitiveIndex;
    vec3 barycentrics;
};

struct ComputeInstance {
    mat4 transform;
    mat4 inverseTransform;
    uint assetIndex, pad1, pad2, pad3; 
};