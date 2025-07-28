#pragma once

#include "Accel.h"
#include "Context.h"
#include "Image.h"
#include "Scene/Scene.h"
#include "Shaders/SharedStructs.h"

class GpuRaytracer {
public:
    GpuRaytracer(Context& context, Scene& scene, uint32_t width, uint32_t height);
    ~GpuRaytracer();

    void render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants);
    void syncWithScene();
    void updateTLAS();
    void updateMeshes();
    void updateTextures();

    const Image& getOutputImage() const { return outputImage; }

private:
    Scene& scene;
    Context& context;
    uint32_t width, height;

    Image outputImage;

    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;

    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;

    vk::StridedDeviceAddressRegionKHR raygenRegion;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;

    Accel tlas;
    Buffer instancesBuffer;
    Buffer meshBuffer;
};