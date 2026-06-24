#pragma once

// Shared CPU/GPU ABI. This file is included by both C++ and Slang.

#define INVALID_INSTANCE 0xFFFFFFFFu
#define InvalidInstance INVALID_INSTANCE
#define GROUP_SIZE 16
#define MAX_LEAF_SIZE 8
#define SAH_BINS 16
#define MAXLEAFSIZE MAX_LEAF_SIZE
#define SAHBINS SAH_BINS

#define CAMERA_PERSPECTIVE  0
#define CAMERA_ORTHOGRAPHIC 1
#define CAMERA_FISHEYE      2
#define CAMERA_THINLENS     3
#define CAMERA_REALISTIC    4
#define MAX_REALISTIC_LENS_ELEMENTS 64
#define MAX_REALISTIC_EXIT_PUPIL_BOUNDS 64

#ifdef __cplusplus
    #include <cstdint>
    #include <limits>
    #include <type_traits>
    #include <glm/vec2.hpp>
    #include <glm/vec3.hpp>
    #include <glm/mat4x4.hpp>

    using namespace glm;

    typedef glm::vec2 float2;
    typedef glm::vec3 float3;
    typedef glm::mat4 float4x4;
    typedef glm::uint uint;
#endif

struct AABB
{
    float3 minBounds;
    float _pad0;
    float3 maxBounds;
    float _pad1;

#ifdef __cplusplus
    AABB()
        : minBounds(std::numeric_limits<float>::max()),
          _pad0(0.0f),
          maxBounds(-std::numeric_limits<float>::max()),
          _pad1(0.0f)
    {}

    void expand(const float3& point) {
        minBounds = glm::min(minBounds, point);
        maxBounds = glm::max(maxBounds, point);
    }

    void expand(const AABB& other) {
        minBounds = glm::min(minBounds, other.minBounds);
        maxBounds = glm::max(maxBounds, other.maxBounds);
    }

    float surfaceArea() const {
        float3 d = maxBounds - minBounds;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    float3 centroid() const {
        return (minBounds + maxBounds) * 0.5f;
    }
#endif
};

struct BVHNode
{
    AABB leftBounds;
    AABB rightBounds;
    uint rightChildOrPrimIndex;
    uint primCount;
    uint splitAxis;
    uint _pad;
};

struct Vertex
{
    float3 position;
    int _pad0;
    float3 normal;
    int _pad1;
    float3 tangent;
    int _pad2;
    float tangentSign;
    float2 uv;
    int _pad3;
};

struct Face
{
    int materialIndex;
};

struct Material
{
    float3 albedo;
    int albedoIndex;
    float specular;
    float metallic;
    float roughness;
    float ior;
    int specularIndex;
    int metallicIndex;
    int roughnessIndex;
    int normalIndex;
    float3 transmissionColor;
    int _pad0;
    float transmission;
    float _pad1;
    float3 emission;
    int _pad2;
    float emissionStrength;
    int _pad3;
    int emissionIndex;
    int transmissionIndex;
    int opacityIndex;
    float opacity;

#ifdef __cplusplus
    Material()
        : albedo{1}, albedoIndex(-1),
          specular(0.5f), metallic(0), roughness(0), ior(1.5f),
          specularIndex(-1), metallicIndex(-1),
          roughnessIndex(-1), normalIndex(-1),
          transmissionColor{1}, _pad0(0),
          transmission(0), _pad1(0),
          emission{1}, _pad2(0),
          emissionStrength(0), _pad3(0),
          emissionIndex(-1), transmissionIndex(-1), opacityIndex(-1), opacity(1)
    {}
#endif
};

struct MeshAddresses
{
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint64_t faceAddress;
    uint64_t materialAddress;
    uint64_t bvhNodeAddress;
    uint64_t bvhIndexAddress;
};

struct DispatchIndirectCommand
{
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct Instance
{
    float4x4 transform;
    float4x4 inverseTransform;
    float4x4 normalTransform;
    uint meshId;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

struct RenderSettings
{
    int samples;
    int diffuseBounces;
    int specularBounces;
    int transmissionBounces;
    int adaptiveSamplingEnabled;
    int adaptiveMinSamples;
    float adaptiveTargetError;
    int russianRouletteStartBounce;
    float exposure;
    int transparentBackground;
    int renderMode;
    int bufferVisualization;
    int taaEnabled;
    int aoSampleAlbedo;
};

struct EnvironmentSettings
{
    int textureIndex;
    int cdfTextureIndex;
    float rotationSin;
    float rotationCos;
    float visibleExposureScale;
    float lightingExposureScale;
    float maxTextureLod;
    int visible;
    float3 directionalDirection;
    float directionalIntensity;
    float rotation;
    float visibleExposure;
    float lightingExposure;
    float directionalSoftAngle;

#ifdef __cplusplus
    EnvironmentSettings()
        : textureIndex(-1), cdfTextureIndex(-1),
          rotationSin(0), rotationCos(1),
          visibleExposureScale(1), lightingExposureScale(1),
          maxTextureLod(0),
          visible(1),
          directionalDirection{0, 1, 0},
          directionalIntensity(0),
          rotation(0), visibleExposure(0), lightingExposure(1),
          directionalSoftAngle(0)
    {}
#endif
};

struct CameraData
{
    float3 position;
    float _pad0;
    float3 direction;
    float _pad1;
    float3 horizontal;
    float _pad2;
    float3 vertical;
    float _pad3;
    float focalLength;
    float focusDistance;
    float aperture;
    float bokehBias;
    int cameraType;
    float orthoHeight;
    float fisheyeFov;
    float nearPlane;
    float farPlane;
};

struct RealisticLensElement
{
    float radius;
    float thickness;
    float apertureRadius;
    float ior;
    float vertexZ;
    int isAperture;
    int _pad0;
    int _pad1;
};

struct RealisticPupilBound
{
    float2 minBounds;
    float2 maxBounds;
};

struct RealisticCameraSettings
{
    RealisticLensElement elements[MAX_REALISTIC_LENS_ELEMENTS];
    RealisticPupilBound pupilBounds[MAX_REALISTIC_EXIT_PUPIL_BOUNDS];
    int elementCount;
    int pupilBoundCount;
    float sensorWidth;
    float sensorHeight;
    float apertureRadius;
    float filmDiagonal;
    float rearElementZ;
    float surfaceOffset;
};

struct SceneSettings
{
    RenderSettings renderSettings;
    EnvironmentSettings environment;
    RealisticCameraSettings realisticCamera;
};

struct PushData
{
    int frame;
    int isMoving;
    int pixelSizePercent;
    int _pushPad0;
    CameraData camera;
};

struct LightGpu
{
    float3 position;
    int type;
    float3 direction;
    int pad0;
    float3 color;
    float intensity;
    float range;
    float innerConeAngle;
    float outerConeAngle;
    float sourceRadius;
    float softSourceRadius;
    float sourceLength;
    float sourceWidth;
    float sourceHeight;
    float lightSourceAngle;
    float lightSourceSoftAngle;
    float lightFalloffExponent;
    int useInverseSquaredFalloff;
    int pad1;
    int pad2;
    int pad3;
    int pad4;
};
