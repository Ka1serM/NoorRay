#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace GaussianRays {

using namespace NoorRHI;

//! The groups of GaussianModel: quaternions are (w, x, y, z) and unnormalized, scales are logarithms and opacities are logits.
struct Parameters {
    Ptr<float> means;
    Ptr<float> featuresDc;
    Ptr<float> featuresRest;
    Ptr<float> opacities;
    Ptr<float> scales;
    Ptr<float> rotations;
};

struct View {
    float4 worldToCamera[3];
    float4 intrinsics;
    float4 position;
    //! The background of the dataset as linear radiance.
    float4 background;
    uint width;
    uint height;
    uint firstRow;
    uint rows;
};

struct Arguments {
    View view;
    Parameters parameters;
    Parameters gradients;
    //! The 59 gradients of each Gaussian in the order mean, quaternion, log scale, opacity logit and 16 SH coefficients.
    AtomicPtr<float> rayGradients;
    AccelerationStructureHandle tlas;
    //! The sRGB-encoded image that the loss sees.
    Ptr<float> image;
    Ptr<float> radiance;
    Ptr<float> imageGradient;
    Ptr<float> gradientSums;
    Ptr<float> gradientCounts;
    uint count;
    uint threads;
    uint shDegree;
    uint accumulate;
};

}
