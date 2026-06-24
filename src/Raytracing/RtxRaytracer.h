#pragma once
#include "WavefrontRaytracer.h"
#include "Vulkan/Accel.h"

// RTX wavefront path tracer using inline RayQuery (VK_KHR_ray_query)
// Set 2 holds the TLAS acceleration structure.
class RtxRaytracer : public WavefrontRaytracer {
    Accel tlas;

    vk::UniqueDescriptorSetLayout tlasSetLayout;
    vk::UniqueDescriptorSet       tlasDescSet;

    void createTlasSet();
    void buildPipelineLayout() override;
    void bindAllDescriptorSets(const vk::CommandBuffer& cmd) const override;

public:
    RtxRaytracer(Scene& scene, uint32_t width, uint32_t height);
    void updateTLAS()     override;
    void updateTextures() override;
};
