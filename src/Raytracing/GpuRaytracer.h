#pragma once

#include <iostream>

#include "Globals.h"
#include "Raytracer.h"
#include "Mesh/MeshAsset.h"
#include "Vulkan/Image.h"
#include "Vulkan/Buffer.h"
#include "Scene/Scene.h"
#include "../Shaders/SharedStructs.h"

class GpuRaytracer : public Raytracer {
protected:
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniqueDescriptorSet descriptorSet;
    vk::UniquePipelineLayout pipelineLayout;

    Buffer instancesBuffer; // RTX: VkAccelerationStructureInstanceKHR, Compute: ComputeInstance
    Buffer meshBuffer;
    
public:

    GpuRaytracer(Scene& scene, const uint32_t width, const uint32_t height): Raytracer(scene, width, height)
    {
    }

    ~GpuRaytracer() override
    {
        context.getDevice().waitIdle();
        std::cout << "Destroying GpuRaytracer" << std::endl;
    }    
    
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
        variableCountAllocInfo.pDescriptorCounts = &MAX_TEXTURES;

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
    
    void updateTextures() override
    {
        std::vector<vk::DescriptorImageInfo> textureImageInfos;
        const auto& textures = scene.getTextures();

        if (textures.empty()) //TODO
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

    void updateMeshes() override
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
        {
            auto emptyMeshAddresses = MeshAddresses{};
            meshBuffer = {context, Buffer::Type::Storage, sizeof(MeshAddresses), &emptyMeshAddresses};
        }
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

