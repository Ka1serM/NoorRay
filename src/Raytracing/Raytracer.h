#pragma once

#include <iostream>

#include "Mesh/MeshAsset.h"
#include "Vulkan/Image.h"
#include "Vulkan/Buffer.h"
#include "Scene/Scene.h"
#include "../Shaders/PathTracing/SharedStructs.h"

class Raytracer {
protected:
    Context& context;
    Scene& scene;
    uint32_t width, height;
    Image outputImage;

    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;
    vk::UniquePipelineLayout pipelineLayout;

    Buffer instancesBuffer; // RTX: VkAccelerationStructureInstanceKHR, Compute: ComputeInstance
    Buffer meshBuffer;
    
public:

    Raytracer(Scene& scene, uint32_t width, uint32_t height): context(scene.getContext()),  // or however you access context
      scene(scene),
      width(width),
      height(height),
      outputImage(scene.getContext(), width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)
    {
    }

    virtual ~Raytracer()
    {
        context.getDevice().waitIdle();
        std::cout << "Destroying ComputeRaytracer" << std::endl;
    }
    
    virtual void render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants) = 0;

    // Common interface methods
    const Image& getOutputImage() const { return outputImage; }
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }

    virtual void updateTLAS() = 0; // Pure virtual since implementations differ
    void updateTextures()
    {
        std::vector<vk::DescriptorImageInfo> textureImageInfos;
        const auto& textures = scene.getTextures();
        textureImageInfos.reserve(textures.size());
        for (const auto& texture : textures)
        {
            vk::DescriptorImageInfo info{};
            info.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
            info.setImageView(texture.getImage().getImageView());
            info.setSampler(texture.getSampler());
            textureImageInfos.push_back(info);
        }    

        uint32_t descriptorCount = static_cast<uint32_t>(textureImageInfos.size());
        vk::WriteDescriptorSet write{descriptorSet.get(),3,0, descriptorCount, vk::DescriptorType::eCombinedImageSampler,textureImageInfos.data()};
        context.getDevice().updateDescriptorSets(write, {});
    }
    void updateMeshes()
    {
        std::vector<MeshAddresses> meshAddresses;
        const auto& meshAssets = scene.getMeshAssets();
        meshAddresses.reserve(meshAssets.size());
        for (const auto& meshAsset : meshAssets)
        {
            meshAsset->updateMaterials();
            meshAddresses.push_back(meshAsset->getBufferAddresses());
        }
    
        meshBuffer = {context, Buffer::Type::Storage, sizeof(MeshAddresses) * meshAddresses.size(), meshAddresses.data()};
        vk::DescriptorBufferInfo bufferInfo = meshBuffer.getDescriptorInfo();
        vk::WriteDescriptorSet write{};
        write.setDstSet(descriptorSet.get());
        write.setDstBinding(2);
        write.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        write.setDescriptorCount(1);
        write.setBufferInfo(bufferInfo);
        context.getDevice().updateDescriptorSets(write, {});
    }
};

