#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Rasterizing {

using namespace NoorRHI;

struct Arguments {
    float4 background;
    Ptr<float4> positions;
    Ptr<uint> triangles;
    Ptr<int> opposites;
    Ptr<float2> uvs;
    Ptr<uint> uvTriangles;
    Ptr<float> texture;
    Ptr<float4> raster;
    Ptr<float> color;
    AtomicPtr<float> image;
    Ptr<float> imageGradient;
    AtomicPtr<float> colorGradient;
    AtomicPtr<float> positionGradient;
    uint width;
    uint height;
    uint triangleCount;
    uint vertexCount;
    uint textureWidth;
    uint textureHeight;
    uint antialiasing;
};

}
