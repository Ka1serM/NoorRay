#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace DirectLighting {

using namespace NoorRHI;

static const uint parameterCount = 9;
static const uint groundInstance = 0;
static const uint objectInstance = 1;
static const uint interiorTerm = 1;
static const uint shadowBoundaryTerm = 2;

struct Arguments {
    float4 cameraOrigin;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    float4 lightCorner;
    float4 lightEdgeU;
    float4 lightEdgeV;
    float4 lightRadiance;
    AccelerationStructureHandle scene;
    Ptr<float> parameters;
    Ptr<float> groundPositions;
    Ptr<uint> groundIndices;
    Ptr<float> objectPositions;
    Ptr<uint> objectIndices;
    Ptr<float> objectNormals;
    Ptr<uint4> objectEdges;
    Ptr<float> weights;
    Ptr<float> image;
    AtomicPtr<float> gradients;
    uint width;
    uint height;
    uint samples;
    uint seed;
    uint edgeCount;
    uint edgeSamples;
    uint terms;
    uint lightSamples;
};

}
