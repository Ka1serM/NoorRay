#pragma once

#include "Raytracer.h"
#include "Vulkan/Buffer.h"
#include "Globals.h"
#include <array>

class WavefrontRaytracer : public Raytracer {
protected:
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet       descriptorSet;

    vk::UniqueDescriptorSetLayout wavefrontSetLayout;
    vk::UniqueDescriptorSet       wavefrontDescSet;

    vk::UniquePipelineLayout pipelineLayout;

    enum Pass { Generate=0, Extend=1, Shade=2, Connect=3, Finalize=4, Advance=5 };
    std::array<vk::UniquePipeline, 6> pipelines;

    int maxShaderBounces = 1;

    Buffer instancesBuffer;
    Buffer meshBuffer;
    Buffer sceneSettingsBuffer;
    Buffer sceneLightsBuffer;

    // Wavefront queue buffers (set 1)
    Buffer countersBuffer;    // 4 ints: activeCount, nextCount, hitCount, shadowCount
    Buffer pathStatesBuffer;  // N × 112 bytes (7 × (float3+uint))
    Buffer rayQueueBuffer;    // N × 32 bytes, bound to both input(2) and output(3) slots
    Buffer hitQueueBuffer;    // N × 48 bytes
    Buffer shadowQueueBuffer; // 2N × 48 bytes

    uint32_t groups2Dx = 0, groups2Dy = 0;
    uint32_t extGroups = 0;

    void createSceneDescriptorSet(const std::vector<vk::DescriptorSetLayoutBinding>& bindings);
    void createWavefrontSet();
    virtual void buildPipelineLayout();

    struct SpvEntry { const void* data; size_t size; };
    void buildPipelines(const std::array<SpvEntry, 6>& spvs);

    void bindOutputImages();
    void writeWavefrontDescriptors();

    void computeBarrier(const vk::CommandBuffer& cmd) const;
    virtual void bindAllDescriptorSets(const vk::CommandBuffer& cmd) const;

    uint32_t maxPixels() const { return width * height; }

    void updateDispatchCounts();

public:
    WavefrontRaytracer(Scene& scene, uint32_t width, uint32_t height);
    ~WavefrontRaytracer() override;

    void render(const vk::CommandBuffer& cmd, const PushData& pc) override;
    void resize(uint32_t newWidth, uint32_t newHeight) override;
    void updateMeshes() override;
    void updateSceneSettings(const SceneSettings& ss) override;

    void updateTLAS()     override = 0;
    void updateTextures() override = 0;
};
