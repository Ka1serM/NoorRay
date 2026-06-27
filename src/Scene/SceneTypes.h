#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda.h>
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

using glm::ivec2;
using glm::mat4;
using glm::quat;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::angleAxis;
using glm::cross;
using glm::dot;
using glm::inverse;
using glm::length;
using glm::mat4_cast;
using glm::normalize;
using glm::perspective;
using glm::quat_cast;
using glm::radians;
using glm::transpose;

static constexpr int CameraPerspective = 0;
static constexpr int CameraOrthographic = 1;
static constexpr int CameraFisheye = 2;
static constexpr int CameraThinLens = 3;
static constexpr int CameraRealistic = 4;
static constexpr int MaxRealisticLensElements = 64;
static constexpr int MaxRealisticExitPupilBounds = 64;

struct Vertex
{
    vec3 position;
    int _pad0;
    vec3 normal;
    int _pad1;
    vec3 tangent;
    int _pad2;
    float tangentSign;
    vec2 uv;
    int _pad3;
};
static_assert(sizeof(Vertex) == 64);

struct Face
{
    int materialIndex;
};

struct Material
{
    vec3 albedo{1.0f};
    int albedoIndex{-1};
    float specular{0.5f};
    float metallic{};
    float roughness{};
    float ior{1.5f};
    int specularIndex{-1};
    int metallicIndex{-1};
    int roughnessIndex{-1};
    int normalIndex{-1};
    vec3 transmissionColor{1.0f};
    int _pad0{};
    float transmission{};
    float _pad1{};
    vec3 emission{1.0f};
    int _pad2{};
    float emissionStrength{};
    int _pad3{};
    int emissionIndex{-1};
    int transmissionIndex{-1};
    int opacityIndex{-1};
    float opacity{1.0f};
};
static_assert(sizeof(Material) == 112);

struct RenderSettings
{
    int samples{};
    int diffuseBounces{};
    int specularBounces{};
    int transmissionBounces{};
    int adaptiveSamplingEnabled{};
    int adaptiveMinSamples{};
    float adaptiveTargetError{};
    int russianRouletteStartBounce{};
    float exposure{};
    int transparentBackground{};
    int renderMode{};
    int bufferVisualization{};
    int taaEnabled{};
    int aoSampleAlbedo{};
};

struct EnvironmentSettings
{
    int textureIndex{-1};
    int cdfTextureIndex{-1};
    float rotationSin{};
    float rotationCos{1.0f};
    float visibleExposureScale{1.0f};
    float lightingExposureScale{1.0f};
    float maxTextureLod{};
    int visible{1};
    vec3 directionalDirection{0.0f, 1.0f, 0.0f};
    float directionalIntensity{};
    float rotation{};
    float visibleExposure{};
    float lightingExposure{1.0f};
    float directionalSoftAngle{};
};

struct CameraData
{
    int cameraType{};
    float focalLength{};
    float focusDistance{};
    float aperture{};
    float bokehBias{};
    float orthoHeight{};
    float fisheyeFov{};
    float nearPlane{};
    float farPlane{};
    float sensorScaleX{};
    float sensorScaleY{};
    int _pad0{};
};

struct RayLutEntry
{
    vec3 origin{};
    float originValid{};
    vec3 direction{0.0f, 0.0f, 1.0f};
    float _pad0{};
};
static_assert(sizeof(RayLutEntry) == 32);

struct RealisticLensElement
{
    float radius{};
    float thickness{};
    float apertureRadius{};
    float ior{};
    float vertexZ{};
    int isAperture{};
    int _pad0{};
    int _pad1{};
};

struct RealisticPupilBound
{
    vec2 minBounds{};
    vec2 maxBounds{};
};

struct RealisticCameraSettings
{
    RealisticLensElement elements[MaxRealisticLensElements]{};
    RealisticPupilBound pupilBounds[MaxRealisticExitPupilBounds]{};
    int elementCount{};
    int pupilBoundCount{};
    float sensorWidth{};
    float sensorHeight{};
    float apertureRadius{};
    float filmDiagonal{};
    float rearElementZ{};
    float surfaceOffset{};
};

struct SceneSettings
{
    RenderSettings renderSettings{};
    EnvironmentSettings environment{};
    RealisticCameraSettings realisticCamera{};
    CameraData camera{};
};

struct PushData
{
    int frame{};
    int isMoving{};
    int pixelSizePercent{};
    int rayLutEnabled{};
    mat4 cameraToWorld{1.0f};
};

struct LightGpu
{
    vec3 position{};
    int type{};
    vec3 direction{};
    int _pad0{};
    vec3 color{};
    float intensity{};
    float range{};
    float innerConeAngle{};
    float outerConeAngle{};
    float sourceRadius{};
    float softSourceRadius{};
    float sourceLength{};
    float sourceWidth{};
    float sourceHeight{};
    float lightSourceAngle{};
    float lightSourceSoftAngle{};
    float lightFalloffExponent{};
    int useInverseSquaredFalloff{};
    int _pad1{};
    int _pad2{};
    int _pad3{};
    int _pad4{};
};
