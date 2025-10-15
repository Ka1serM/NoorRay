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
    Buffer assetBuffer;
    
public:

    GpuRaytracer(Scene& scene, const uint32_t width, const uint32_t height): Raytracer(scene, width, height)
    {
    }

    ~GpuRaytracer() override
    {
        context.getDevice().waitIdle();
        LOG_INFO( "Destroying GpuRaytracer");
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

        // Create binding flags info structure
        vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        bindingFlagsInfo.setBindingFlags(bindingFlags);

        // Create descriptor set layout
        vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
        descSetLayoutInfo.setBindings(bindings);
        descSetLayoutInfo.setPNext(&bindingFlagsInfo);
        descSetLayoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);

        descSetLayout = context.getDevice().createDescriptorSetLayoutUnique(descSetLayoutInfo);

        // Allocate descriptor set with variable descriptor count
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
        colorInfo.setImageView(outputColor.getView());
        colorInfo.setImageLayout(vk::ImageLayout::eGeneral);
        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(1)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(colorInfo));

        // Output Albedo (binding = 2)
        vk::DescriptorImageInfo albedoInfo{};
        albedoInfo.setImageView(outputAlbedo.getView());
        albedoInfo.setImageLayout(vk::ImageLayout::eGeneral);
        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(2)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(albedoInfo));

        // Output Normal (binding = 3)
        vk::DescriptorImageInfo normalInfo{};
        normalInfo.setImageView(outputNormal.getView());
        normalInfo.setImageLayout(vk::ImageLayout::eGeneral);
        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(3)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(normalInfo));

        // Output Crypto/Object ID (binding = 4)
        vk::DescriptorImageInfo cryptoInfo{};
        cryptoInfo.setImageView(outputCrypto.getView());
        cryptoInfo.setImageLayout(vk::ImageLayout::eGeneral);
        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                                   .setDstSet(descriptorSet.get())
                                   .setDstBinding(4)
                                   .setDescriptorType(vk::DescriptorType::eStorageImage)
                                   .setDescriptorCount(1)
                                   .setImageInfo(cryptoInfo));


        // Output Position (binding = 5)
        vk::DescriptorImageInfo positionInfo{};
        positionInfo.setImageView(outputPosition.getView());
        positionInfo.setImageLayout(vk::ImageLayout::eGeneral);
        descriptorWrites.push_back(vk::WriteDescriptorSet{}
                           .setDstSet(descriptorSet.get())
                           .setDstBinding(5)
                           .setDescriptorType(vk::DescriptorType::eStorageImage)
                           .setDescriptorCount(1)
                           .setImageInfo(positionInfo));
        
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
            info.setImageView(texture.getImage().getView());
            info.setSampler(texture.getSampler());
            textureImageInfos.push_back(info);
        }

        const uint32_t descriptorCount = static_cast<uint32_t>(textureImageInfos.size());

        const vk::WriteDescriptorSet write{
            descriptorSet.get(),
            7, // DstBinding 7 is for textures
            0,
            descriptorCount,
            vk::DescriptorType::eCombinedImageSampler,
            textureImageInfos.data()
        };
        context.getDevice().updateDescriptorSets(write, {});
    }

    void updateMeshes() override
    {
        std::vector<AssetData> assetDatas;
        const auto& meshAssets = scene.getMeshAssets();
        for (const auto& meshAsset : meshAssets)
        {
            meshAsset->updateMaterials();
            AssetData data{};
            data.bufferAddress = meshAsset->getAddressBuffer().getDeviceAddress();
            assetDatas.push_back(data);
        }

        if (assetDatas.empty())
        {
            auto emptyAssetData = AssetData{};
            assetBuffer = {context, Buffer::Type::Storage, sizeof(AssetData), &emptyAssetData};
        }
        else
            assetBuffer = {context, Buffer::Type::Storage, sizeof(AssetData) * assetDatas.size(), assetDatas.data()};

        vk::DescriptorBufferInfo bufferInfo = assetBuffer.getDescriptorInfo();

        vk::WriteDescriptorSet write{};
        write.setDstSet(descriptorSet.get());
        write.setDstBinding(6); // DstBinding 6 is for assets
        write.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        write.setDescriptorCount(1); // Always 1 for a single buffer binding
        write.setBufferInfo(bufferInfo); 

        context.getDevice().updateDescriptorSets(write, {});
    }

};

