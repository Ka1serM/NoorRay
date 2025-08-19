#pragma once

#include "GpuRaytracer.h"

class ComputeRaytracer : public GpuRaytracer {
public:
    
    void render(const vk::CommandBuffer& commandBuffer, const PushConstantsData& pushConstants) override;
    void updateTLAS() override;
    ComputeRaytracer(Scene& scene, uint32_t width, uint32_t height);
};