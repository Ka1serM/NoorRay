#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace GaussianPaths {

using namespace NoorRHI;

//! The floats of one Gaussian; a buffer holds the means of all Gaussians first, then the quaternions (w, x, y, z), the scales, the opacities and the colours.
static const uint parameterStride = 14;
static const uint quaternionOffset = 3;
static const uint scaleOffset = 7;
static const uint opacityOffset = 10;
static const uint colorOffset = 11;

struct Arguments {
    //! The w component is the tangent of half the field of view.
    float4 cameraOrigin;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    //! The position and the radiant intensity of the point light.
    float4 light;
    //! The linear sRGB colour of the environment and the colour temperature of the point light.
    float4 environment;
    AccelerationStructureHandle proxies;
    AccelerationStructureHandle mesh;
    Ptr<float> parameters;
    Ptr<float> positions;
    Ptr<uint> indices;
    //! The linear sRGB albedo of each triangle.
    Ptr<float> albedos;
    Ptr<float4> spectralEntries;
    Ptr<float> weights;
    Ptr<float> image;
    AtomicPtr<float> gradients;
    uint count;
    uint width;
    uint height;
    uint firstSample;
    uint sampleEnd;
    uint samples;
    uint seed;
    //! Zero renders the sRGB channels of the path tracer's RGB mode in place of four wavelengths.
    uint spectral;
    float filterWidth;
};

}
