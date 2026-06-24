#pragma once
#include "WavefrontRaytracer.h"

// Software-BVH wavefront path tracer (no RT hardware required)
class ComputeRaytracer : public WavefrontRaytracer {
public:
    ComputeRaytracer(Scene& scene, uint32_t width, uint32_t height);
    void updateTLAS()     override;
    void updateTextures() override;
};
