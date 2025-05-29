#pragma once

#include <memory>

#include "Accel.h"
#include "Buffer.h"
#include "Context.h"
#include "Texture.h"
#include "Shaders/SharedStructs.h"

template<class>
inline constexpr bool always_false = false;

class Renderer {

private:
    bool dirty;

    Context& context;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;

    std::vector<MeshAddresses> meshAddresses;

    std::vector<PointLight> pointLights;
    Buffer materialBuffer;
    Buffer pointLightBuffer;
    Buffer textureBuffer;
    std::vector<vk::DescriptorImageInfo> textureImageInfos;
    Buffer meshBuffer;
    Buffer instancesBuffer;

    std::vector<const vk::AccelerationStructureInstanceKHR*> instancePtrs;

    template <class T>
    void updateStorageBuffer(uint32_t binding, const std::vector<T>& data, Buffer& buffer);
    void updateTextureDescriptors(const std::vector<Texture>& textures_);

public:
    std::vector<Material> materials;
    std::vector<Texture> textures;

    Renderer(Context& context);

    void buildTLAS();

    void render(uint32_t imageIndex, const PushConstants& pushConstants);

    void updateStorageImage(vk::ImageView storageImageView);

    vk::CommandBuffer getCommandBuffer(uint32_t imageIndex);
    vk::SwapchainKHR getSwapChain() const;
    std::vector<vk::Image>& getSwapchainImages();

    void add(Texture&& element);
    void add(const PointLight& element);
    void add(const MeshAddresses& element);
    void add(const Material& element);
    void add(vk::AccelerationStructureInstanceKHR& element);

    void add(std::vector<Texture>&& elements);
    void add(const std::vector<PointLight>& elements);
    void add(const std::vector<MeshAddresses>& elements);
    void add(const std::vector<Material>& elements);
    void add(const std::vector<const vk::AccelerationStructureInstanceKHR*>& elements);

    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

    std::vector<vk::UniqueShaderModule> shaderModules;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueSemaphore imageAcquiredSemaphore;

    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;

    vk::StridedDeviceAddressRegionKHR raygenRegion;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;

    Accel tlas;

    void markDirty() { dirty = true; };
    void resetDirty() { dirty = false; };
    bool getDirty() const { return dirty; }
};
