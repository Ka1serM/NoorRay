#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"

NR_NAMESPACE_BEGIN

static const uint spectralTableSize = 1024;
static const uint conductorSampleCount = 16;
static const float conductorLambdaMin = 380.0;
static const float conductorLambdaStep = 25.0;

//! Entry i is the wavelength of the hero sample u = (i + 0.5) / spectralTableSize. film is the colour matching weight divided by the wavelength pdf.
struct SpectralEntry {
    float4 basisAndLambda;
    float4 filmAndPdf;
};

//! Complex IOR samples of one metal, conductorSampleCount per metal from conductorLambdaMin.
struct ConductorSample {
    float eta;
    float k;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::SpectralEntry) == 32);
static_assert(sizeof(NoorRay::Shaders::ConductorSample) == 8);
#endif
