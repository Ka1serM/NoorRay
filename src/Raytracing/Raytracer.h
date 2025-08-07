#pragma once

#include <iostream>

#include "Globals.h"
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
    Image outputColor;
    Image outputAlbedo;
    Image outputNormal;
    Image outputCrypto;

    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;
    vk::UniquePipelineLayout pipelineLayout;

    Buffer instancesBuffer; // RTX: VkAccelerationStructureInstanceKHR, Compute: ComputeInstance
    Buffer meshBuffer;
    
public:

    Raytracer(Scene& scene, uint32_t width, uint32_t height): context(scene.getContext()),
      scene(scene),
      width(width),
      height(height),
      outputColor(scene.getContext(), width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputAlbedo(scene.getContext(), width, height, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputNormal(scene.getContext(), width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputCrypto(scene.getContext(), width, height, vk::Format::eR32Uint, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)
    {
    }

    virtual ~Raytracer()
    {
        context.getDevice().waitIdle();
        std::cout << "Destroying ComputeRaytracer" << std::endl;
    }
    
    virtual void render(const vk::CommandBuffer& commandBuffer, const PushConstantsData& pushConstants) = 0;

    Image& getOutputColor() { return outputColor; }
    Image& getOutputAlbedo() { return outputAlbedo; }
    Image& getOutputNormal() { return outputNormal; }
    Image& getOutputCrypto() { return outputCrypto; }
    
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }

    virtual void updateTLAS() = 0; // Pure virtual since implementations differ

    void createDescriptorSet(std::vector<vk::DescriptorSetLayoutBinding> bindings)
    {
        assert(!bindings.empty() && "Bindings vector must not be empty");

        std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size());
        const uint32_t variableBindingIndex = static_cast<uint32_t>(bindings.size() - 1);

        // Apply flags only to the last binding
        bindingFlags[variableBindingIndex] =
            vk::DescriptorBindingFlagBits::ePartiallyBound |
            vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
            vk::DescriptorBindingFlagBits::eUpdateAfterBind;

        // Make sure descriptorCount is set correctly
        bindings[variableBindingIndex].descriptorCount = MAX_TEXTURES;

        // Step 2: Create binding flags info structure
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        bindingFlagsInfo.setBindingFlags(bindingFlags);

        // Step 3: Create descriptor set layout
        vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
        descSetLayoutInfo.setBindings(bindings);
        descSetLayoutInfo.setPNext(&bindingFlagsInfo);
        descSetLayoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);

        descSetLayout = context.getDevice().createDescriptorSetLayoutUnique(descSetLayoutInfo);

        // Step 4: Allocate descriptor set with variable descriptor count
        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountAllocInfo{};
        variableCountAllocInfo.descriptorSetCount = 1;
        uint32_t variableCount = MAX_TEXTURES;
        variableCountAllocInfo.pDescriptorCounts = &variableCount;

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.setDescriptorPool(context.getDescriptorPool());
        allocInfo.setSetLayouts(descSetLayout.get());
        allocInfo.setDescriptorSetCount(1);
        allocInfo.setPNext(&variableCountAllocInfo);

        descriptorSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());
    }
    
    void bindOutputImages()
    {
        std::vector<vk::WriteDescriptorSet> descriptorWrites;

        // Output Color (binding = 1)
        vk::DescriptorImageInfo colorInfo{};
        colorInfo.setImageView(outputColor.getImageView());
        colorInfo.setImageLayout(vk::ImageLayout::eGeneral);

        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(1)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(colorInfo));

        // Output Albedo (binding = 2)
        vk::DescriptorImageInfo albedoInfo{};
        albedoInfo.setImageView(outputAlbedo.getImageView());
        albedoInfo.setImageLayout(vk::ImageLayout::eGeneral);

        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(2)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(albedoInfo));

        // Output Normal (binding = 3)
        vk::DescriptorImageInfo normalInfo{};
        normalInfo.setImageView(outputNormal.getImageView());
        normalInfo.setImageLayout(vk::ImageLayout::eGeneral);

        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(3)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(normalInfo));

        // Output Crypto/Object ID (binding = 4)
        vk::DescriptorImageInfo cryptoInfo{};
        cryptoInfo.setImageView(outputCrypto.getImageView());
        cryptoInfo.setImageLayout(vk::ImageLayout::eGeneral);

        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(4)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(cryptoInfo));

        // Update all descriptor sets at once
        context.getDevice().updateDescriptorSets(descriptorWrites, {});
    }
    
    void updateTextures()
    {
        std::vector<vk::DescriptorImageInfo> textureImageInfos;
        const auto& textures = scene.getTextures();

        if (textures.empty())
            return;
  
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

        const vk::WriteDescriptorSet write{
            descriptorSet.get(),
            6, // DstBinding 6 is for textures
            0,
            descriptorCount,
            vk::DescriptorType::eCombinedImageSampler,
            textureImageInfos.data()
        };
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

        if (meshAddresses.empty())
            meshBuffer = {context, Buffer::Type::Storage, sizeof(MeshAddresses), {}};
        else
            meshBuffer = {context, Buffer::Type::Storage, sizeof(MeshAddresses) * meshAddresses.size(), meshAddresses.data()};

        vk::DescriptorBufferInfo bufferInfo = meshBuffer.getDescriptorInfo();

        vk::WriteDescriptorSet write{};
        write.setDstSet(descriptorSet.get());
        write.setDstBinding(5); // DstBinding 5 is for meshes
        write.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        write.setDescriptorCount(1); // Always 1 for a single buffer binding
        write.setBufferInfo(bufferInfo); 

        context.getDevice().updateDescriptorSets(write, {});
    }

};

