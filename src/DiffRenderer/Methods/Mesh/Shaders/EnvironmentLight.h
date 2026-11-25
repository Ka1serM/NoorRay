#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace EnvironmentLight {

using namespace NoorRHI;

//! The mip levels of the cube map from the base size down to 16, as nvdiffrec builds them.
static const uint levelCount = 6;
static const float minimumRoughness = 0.08;
static const float maximumRoughness = 0.5;

//! A trainable cube map with split-sum prefiltering (Munkberg et al. 2022); every level holds 6 faces of size^2 RGB texels, face-major, rows from the top.
struct Light {
    //! The box-filtered levels; the first level is the trained base.
    Ptr<float> levels;
    //! Each level filtered with the GGX lobe of its roughness: the normalized RGB and the sum of the weights.
    Ptr<float4> specular;
    //! The cosine-filtered last level, as the irradiance over pi.
    Ptr<float> diffuse;
    AtomicPtr<float> specularGradient;
    AtomicPtr<float> diffuseGradient;
    //! The gradient of every level; the first level is the gradient of the base.
    Ptr<float> levelGradient;
    //! The cosine of the cone that holds 99% of the GGX lobe of each level.
    float4 cutoffLow;
    float2 cutoffHigh;
    uint baseSize;
    //! The level that a per-level pass works on.
    uint level;
};

struct Arguments {
    Light light;
    //! The factor of the base gradient; nvdiffrec scales it by 64.
    float gradientScale;
    //! The weight of the white balance term.
    float whiteWeight;
    uint padding0;
    uint padding1;
};

}
