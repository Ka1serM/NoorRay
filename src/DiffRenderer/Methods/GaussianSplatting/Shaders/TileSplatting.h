#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace TileSplatting {

using namespace NoorRHI;

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
    float4 background;
    uint width;
    uint height;
    uint tilesX;
    uint tilesY;
};

struct Arguments {
    View view;
    Parameters parameters;
    //! dL/d of every parameter; preprocessBackward writes it and GpuAdam updates each group from it.
    Parameters gradients;
    //! The pixel mean in the pixel-centre-at-integer convention of the CUDA rasterizer, the depth and the radius.
    Ptr<float4> projected;
    Ptr<float4> conics;
    Ptr<float> colors;
    Ptr<uint> tilesTouched;
    Ptr<uint> offsets;
    //! Pairs of words, the depth bits low and the tile high.
    Ptr<uint> keys;
    Ptr<uint> values;
    Ptr<uint> instanceCount;
    Ptr<uint> instanceTotal;
    Ptr<uint> ranges;
    Ptr<float> image;
    Ptr<float> finalTransmittance;
    Ptr<uint> lastContributors;
    Ptr<float> imageGradient;
    //! dL/dmean2D in pixels, dL/dconic, dL/dopacity and dL/dcolor of each Gaussian.
    AtomicPtr<float> splatGradients;
    Ptr<float> gradientSums;
    Ptr<float> gradientCounts;
    uint count;
    uint threads;
    uint instanceCapacity;
    uint shDegree;
    uint accumulate;
};

struct DensifyArguments {
    Parameters parameters;
    Ptr<float> gradientSums;
    Ptr<float> gradientCounts;
    Ptr<uint> outputs;
    Ptr<uint> offsets;
    Ptr<uint> newCount;
    Ptr<uint> parameterSources;
    Ptr<uint> momentSources;
    Ptr<uint> children;
    Ptr<float> destination;
    Ptr<float> firstMoments;
    Ptr<float> secondMoments;
    uint count;
    uint threads;
    float gradientThreshold;
    float splitScale;
    float pruneOpacity;
    //! The largest scale that survives the prune, or infinity before the size prune starts.
    float pruneScale;
    float resetOpacity;
    uint seed;
};

}
