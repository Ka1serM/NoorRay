#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace Smoothing {

using namespace NoorRHI;

//! Solves (I + lambda L) x = right by conjugate gradients on three columns; each pass sums the partial dot products of the last.
struct SolveArguments {
    Ptr<uint> offsets;
    Ptr<uint> neighbors;
    Ptr<float> right;
    Ptr<float> solution;
    Ptr<float> residual;
    Ptr<float> direction;
    Ptr<float> product;
    Ptr<float> products;
    Ptr<float> residuals;
    Ptr<float> nextResiduals;
    uint vertexCount;
    uint groupCount;
    float lambda;
};

struct AdamArguments {
    Ptr<float> gradients;
    Ptr<float> first;
    Ptr<float> second;
    Ptr<float> step;
    AtomicPtr<uint> largestSecond;
    uint count;
    float beta1;
    float beta2;
    float stepSize;
    float secondCorrection;
    float epsilon;
};

struct ApplyArguments {
    Ptr<float> positions;
    Ptr<float> offsets;
    uint count;
};

}
