#pragma once

#include "EnvironmentLight.h"
#include "NoorRHI/Shaders/ShaderTypes.h"

namespace MeshRasterizing {

using namespace NoorRHI;

//! kd in rgb, then ks as occlusion, roughness and metallic, as the materials of nvdiffrec.
static const uint vertexChannels = 6;
//! kd, ks and the tangent space normal of the normal map.
static const uint texelChannels = 9;
static const uint splitSumSize = 256;

//! Maps OpenCV camera space to Vulkan clip space with y down and the depth in [0, 1].
struct Projection {
    float4 row0;
    float4 row1;
    float4 row2;
    //! The x and y scales and offsets of the clip coordinates relative to the camera depth.
    float4 lens;
    //! The scale and offset of the clip depth.
    float2 depth;
    uint width;
    uint height;
};

struct Arguments {
    Projection projection;
    float4 background;
    float4 cameraPosition;
    EnvironmentLight::Light light;
    Ptr<float> positions;
    Ptr<float> vertexMaterials;
    Ptr<uint> triangles;
    Ptr<int> opposites;
    Ptr<float4> clip;
    TextureHandle ids;
    Ptr<float4> raster;
    Ptr<float> shaded;
    Ptr<float> image;
    Ptr<float> imageGradient;
    Ptr<float> shadedGradient;
    AtomicPtr<float> clipGradient;
    AtomicPtr<float> vertexMaterialAccumulator;
    Ptr<float> positionGradient;
    Ptr<float> vertexMaterialGradient;
    //! The sums of the face normals and of the face tangents of each vertex, and the normalized results.
    AtomicPtr<float> normalSums;
    AtomicPtr<float> tangentSums;
    Ptr<float> normals;
    Ptr<float> tangents;
    AtomicPtr<float> normalGradient;
    //! The gradient of the sums of the face normals, before the cross products.
    Ptr<float> normalSumGradient;
    //! The position gradient through the vertex normals.
    AtomicPtr<float> shadingPositionGradient;
    Ptr<float2> uvs;
    Ptr<uint> uvTriangles;
    Ptr<float> texels;
    AtomicPtr<float> texelAccumulator;
    Ptr<float> texelGradient;
    //! The split-sum terms of the GGX BRDF, x for NdotV and y for the roughness: the scale and the bias of F0.
    Ptr<float2> splitSum;
    uint vertexCount;
    uint triangleCount;
    uint antialiasing;
    //! 0 shades with the vertex materials.
    uint textureWidth;
    uint textureHeight;
    uint seed;
    //! The weights of the kd smoothness and the occlusion terms of nvdiffrec, already divided by the pixel count.
    float smoothWeight;
    float occlusionWeight;
    float minimumRoughness;
    uint padding0;
    uint padding1;
    uint padding2;
};

}
