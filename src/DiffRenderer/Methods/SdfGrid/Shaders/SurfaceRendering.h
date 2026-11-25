#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"

namespace SurfaceRendering {

using namespace NoorRHI;

static const uint shCoefficients = 9;
static const uint colourChannels = 27;

//! Values on the nodes origin + voxel * (x, y, z), x fastest; trilinear between the nodes.
struct Grid {
    //! The first node in xyz and the voxel size in w.
    float4 origin;
    //! The node counts in xyz and their product in w.
    uint4 size;
};

struct Arguments {
    //! The rows of the camera-to-world rotation transposed: the world directions of the camera x, y and z axes.
    float4 cameraRight;
    float4 cameraDown;
    float4 cameraForward;
    float4 cameraPosition;
    //! fx, fy, cx and cy in pixels.
    float4 intrinsics;
    //! The constant background in rgb; w is 1 when the environment map replaces it.
    float4 background;
    Grid grid;
    //! The grid that resample reads.
    Grid source;
    Ptr<float> sdf;
    //! 27 values for each node: the degree-2 SH coefficients, coefficient-major, rgb inside.
    Ptr<float> colour;
    //! A latitude-longitude RGB map in logits, environmentWidth x environmentHeight.
    Ptr<float> environment;
    AtomicPtr<float> sdfGradient;
    AtomicPtr<float> colourGradient;
    AtomicPtr<float> environmentGradient;
    Ptr<float> sourceSdf;
    Ptr<float> sourceColour;
    //! For each node the closest point of the zero level set that the jump flooding found so far, w is 0 for none.
    Ptr<float4> seeds;
    Ptr<float4> nextSeeds;
    //! RGB for each pixel, rows from the top.
    Ptr<float> image;
    Ptr<float> imageGradient;
    //! The RGBA8 train image, for the rays that see only the background.
    Ptr<uint> target;
    Ptr<float> clearTarget;
    uint width;
    uint height;
    uint seed;
    uint environmentWidth;
    uint environmentHeight;
    uint count;
    uint threads;
    //! The SH coefficients in use: 1, 4 or 9; view dependence starts late so that it cannot paint the scene onto the start sphere.
    uint activeCoefficients;
    //! Each pixel takes samplesPerSide^2 stratified jittered rays.
    uint samplesPerSide;
    //! 1 adds the relaxed boundary term of the silhouettes.
    uint boundaryTerm;
    //! The node offset of the current jump flooding pass.
    uint jump;
    //! The smallest sphere tracing step, in world units.
    float step;
    float curvatureWeight;
    //! The weight that pushes the surface away from rays that see only background; a white surface before a white background has no other gradient.
    float emptyWeight;
    //! The width of the relaxed silhouette band in pixels; the SDF threshold of a ray is this width times the footprint of one pixel at its distance.
    float bandPixels;
    float padding;
};

}
