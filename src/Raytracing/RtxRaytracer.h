#pragma once

#include "Raytracer.h"
#include "Vulkan/Accel.h"

class RtxRaytracer : public Raytracer {
    Accel tlas;
    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;
    
    vk::StridedDeviceAddressRegionKHR raygenRegion;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;

public:
    RtxRaytracer(Scene& scene, uint32_t width, uint32_t height);

    void render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants) override;

    void updateTLAS() override;
};