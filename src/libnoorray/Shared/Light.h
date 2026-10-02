#pragma once

#include "Types.h"

// Keep these values aligned with Unreal's ELightUnits.
static const uint LightUnitsUnitless = 0u;
static const uint LightUnitsCandelas = 1u;
static const uint LightUnitsLumens = 2u;
static const uint LightUnitsEV = 3u;
static const uint LightUnitsNits = 4u;

// Unreal's per-light controls, as its path tracer applies them.
struct LightControls
{
#ifdef __cplusplus
    LightControls()
        : castShadows(1u), lightingChannels(1u), diffuseScale(1.0f),
          specularScale(1.0f), indirectLightingIntensity(1.0f), visible(1u)
    {
    }
#endif
    uint castShadows;
    // Bit n is Unreal's lighting channel n; a light only reaches surfaces
    // that share a channel with it.
    uint lightingChannels;
    float diffuseScale;
    float specularScale;
    // Scales the light reaching a path after a rough bounce.
    float indirectLightingIntensity;
    // A hidden light keeps its record and selection weight but lights nothing.
    uint visible;
};

// How a local light fades with distance, as Unreal's local lights do: an
// inverse square falloff windowed to zero at the attenuation radius, or the
// legacy exponent falloff that replaces the inverse square law.
struct LightFalloff
{
#ifdef __cplusplus
    LightFalloff() : invRadius(0.0f), exponent(8.0f), inverseSquared(1u) {}
#endif
    // One over the attenuation radius; 0 disables the window.
    float invRadius;
    float exponent;
    uint inverseSquared;
};

// These records intentionally mirror the concrete light families instead of
// relying on a tagged record with an untyped parameter array. The same records
// are used by scene authoring, typed GPU buffers, and Slang. Lengths are in
// world units. The explicit padding keeps every structured-buffer stride a
// multiple of 16 bytes.
struct PointLight
{
#ifdef __cplusplus
    PointLight()
        : position{}, softRadius{}, color(1.0f), intensity(1.0f),
          axis(1.0f, 0.0f, 0.0f), sourceLength{}, selectionWeight{},
          units(LightUnitsUnitless), falloff{}, controls{}, padding{}
    {
    }
#endif
    float3 position;
    // The sphere's radius, or the capsule's.
    float softRadius;
    float3 color;
    float intensity;
    // A capsule's axis; it extends sourceLength / 2 either way along it.
    float3 axis;
    float sourceLength;
    float selectionWeight;
    uint units;
    LightFalloff falloff;
    LightControls controls;
    float padding;
};

struct SpotLight
{
#ifdef __cplusplus
    SpotLight()
        : position{}, softRadius{}, direction{}, innerConeAngle(20.0f),
          color(1.0f), intensity(1.0f), axis(1.0f, 0.0f, 0.0f), sourceLength{},
          outerConeAngle(30.0f), selectionWeight{}, units(LightUnitsUnitless),
          falloff{}, controls{}
    {
    }
#endif
    float3 position;
    float softRadius;
    float3 direction;
    float innerConeAngle;
    float3 color;
    float intensity;
    float3 axis;
    float sourceLength;
    float outerConeAngle;
    float selectionWeight;
    uint units;
    LightFalloff falloff;
    LightControls controls;
};

struct RectLight
{
#ifdef __cplusplus
    RectLight()
        : position{}, width(1.0f), direction{}, height(1.0f),
          tangent(1.0f, 0.0f, 0.0f), twoSided{}, color(1.0f),
          intensity(1.0f), barnDoorAngle(90.0f), barnDoorLength{},
          selectionWeight{}, units(LightUnitsUnitless), falloff{}, controls{},
          padding{}
    {
    }
#endif
    float3 position;
    float width;
    float3 direction;
    float height;
    float3 tangent;
    uint twoSided;
    float3 color;
    float intensity;
    float barnDoorAngle;
    float barnDoorLength;
    float selectionWeight;
    uint units;
    LightFalloff falloff;
    LightControls controls;
    float padding[3];
};

struct DirectionalLight
{
#ifdef __cplusplus
    DirectionalLight()
        : direction(0.0f, -1.0f, 0.0f), softAngle(0.53f), color(1.0f),
          intensity(1.0f), selectionWeight{}, units(LightUnitsUnitless), controls{}
    {
    }
#endif
    float3 direction;
    // The full angle of the sun's disk, in degrees.
    float softAngle;
    float3 color;
    float intensity;
    float selectionWeight;
    uint units;
    LightControls controls;
};

struct MeshLight
{
    float3 a;
    uint instanceIndex;
    float3 b;
    uint primitiveIndex;
    float3 c;
    float area;
    float selectionWeight;
    float padding[3];
};

#ifdef __cplusplus
static_assert(sizeof(PointLight) == 96 && sizeof(SpotLight) == 112 && sizeof(RectLight) == 128
    && sizeof(DirectionalLight) == 64);

namespace nr::graphics {
using ::PointLight;
using ::SpotLight;
using ::RectLight;
using ::DirectionalLight;
using ::MeshLight;
} // namespace nr::graphics
#endif
