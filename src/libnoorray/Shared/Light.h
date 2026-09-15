#pragma once

#include "Types.h"

// These records intentionally mirror the concrete light families instead of
// relying on a tagged record with an untyped parameter array. The same records
// are used by scene authoring, typed GPU buffers, and Slang. The explicit
// padding keeps every structured-buffer stride a multiple of 16 bytes.
struct PointLight
{
#ifdef __cplusplus
    PointLight()
        : position{}, softRadius{}, color(1.0f), intensity(1.0f),
          selectionWeight{}, padding{}
    {
    }
#endif
    float3 position;
    float softRadius;
    float3 color;
    float intensity;
    float selectionWeight;
    float padding[3];
};

struct SpotLight
{
#ifdef __cplusplus
    SpotLight()
        : position{}, softRadius{}, direction{}, innerConeAngle(20.0f),
          color(1.0f), intensity(1.0f), outerConeAngle(30.0f),
          selectionWeight{}, padding{}
    {
    }
#endif
    float3 position;
    float softRadius;
    float3 direction;
    float innerConeAngle;
    float3 color;
    float intensity;
    float outerConeAngle;
    float selectionWeight;
    float padding[2];
};

struct RectLight
{
#ifdef __cplusplus
    RectLight()
        : position{}, width(1.0f), direction{}, height(1.0f),
          tangent(1.0f, 0.0f, 0.0f), twoSided{}, color(1.0f),
          intensity(1.0f), barnDoorAngle(90.0f), barnDoorLength{},
          selectionWeight{}, padding{}
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
    float padding;
};

struct DirectionalLight
{
#ifdef __cplusplus
    DirectionalLight()
        : direction(0.0f, -1.0f, 0.0f), softAngle(0.53f), color(1.0f),
          intensity(1.0f), selectionWeight{}, padding{}
    {
    }
#endif
    float3 direction;
    float softAngle;
    float3 color;
    float intensity;
    float selectionWeight;
    float padding[3];
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
namespace nr::graphics {
using ::PointLight;
using ::SpotLight;
using ::RectLight;
using ::DirectionalLight;
using ::MeshLight;
} // namespace nr::graphics
#endif
