#pragma once

#include "Raytracer.h"

class ComputeRaytracer : public Raytracer {
public:
    
    void render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants) override;
    void updateTLAS() override;
    ComputeRaytracer(Scene& scene, uint32_t width, uint32_t height);
};