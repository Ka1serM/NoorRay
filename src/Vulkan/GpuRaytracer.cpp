#include "GpuRaytracer.h"

#include <iostream>
#include <ranges>
#include <algorithm>

#include "Globals.h"
#include "Utils.h"
#include "Scene/MeshInstance.h"
#include "Vulkan/Image.h"

GpuRaytracer::GpuRaytracer(Context& context, Scene& scene, uint32_t width, uint32_t height)
    : scene(scene), context(context), width(width), height(height), 
      outputImage(context, width, height, vk::Format::eR32G32B32A32Sfloat,
                  vk::ImageUsageFlagBits::eStorage |
                  vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled)
{
    static constexpr unsigned char RayGeneration[] = {
        #embed "../shaders/PathTracing/RayGeneration.spv"
    };
    static constexpr unsigned char PathTracingMiss[] = {
        #embed "../shaders/PathTracing/Miss.spv"
    };
    static constexpr unsigned char PathTracingClosestHit[] = {
        #embed "../shaders/PathTracing/ClosestHit.spv"
    };

    constexpr const unsigned char* shaders[] = {
        RayGeneration,
        PathTracingMiss,
        PathTracingClosestHit,
    };

    constexpr size_t shaderSizes[] = {
        sizeof(RayGeneration),
        sizeof(PathTracingMiss),
        sizeof(PathTracingClosestHit),
    };

    constexpr vk::ShaderStageFlagBits shaderStages[] = {
        vk::ShaderStageFlagBits::eRaygenKHR,
        vk::ShaderStageFlagBits::eMissKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR,
    };

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesVector;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    uint32_t raygenCount = 0;
    uint32_t missCount = 0;
    uint32_t hitCount = 0;

    for (size_t i = 0; i < std::size(shaders); ++i)
    {
        shaderModules.emplace_back(context.getDevice().createShaderModuleUnique({
            {}, shaderSizes[i], reinterpret_cast<const uint32_t*>(shaders[i])
        }));

        shaderStagesVector.push_back({{}, shaderStages[i], *shaderModules.back(), "main"});

        if (shaderStages[i] == vk::ShaderStageFlagBits::eRaygenKHR)
        {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            raygenCount++;
        }
        else if (shaderStages[i] == vk::ShaderStageFlagBits::eMissKHR)
        {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            missCount++;
        }
        else if (shaderStages[i] == vk::ShaderStageFlagBits::eClosestHitKHR)
        {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            hitCount++;
        }
    }

    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},
        {3, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR}
    };

    std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(), vk::DescriptorBindingFlags{});
    bindingFlags[3] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.setBindingFlags(bindingFlags);

    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
    descSetLayoutInfo.setBindings(bindings);
    descSetLayoutInfo.setPNext(&bindingFlagsInfo);
    descSetLayout = context.getDevice().createDescriptorSetLayoutUnique(descSetLayoutInfo);

    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountAllocInfo{};
    variableCountAllocInfo.setDescriptorCounts(MAX_TEXTURES);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.getDescriptorPool());
    allocInfo.setSetLayouts(descSetLayout.get());
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPNext(&variableCountAllocInfo);

    descriptorSet = std::move(context.getDevice().allocateDescriptorSetsUnique(allocInfo).front());

    vk::PushConstantRange pushRange{};
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstants));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(descSetLayout.get());
    pipelineLayoutInfo.setPushConstantRanges(pushRange);

    pipelineLayout = context.getDevice().createPipelineLayoutUnique(pipelineLayoutInfo);

    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo{};
    rtPipelineInfo.setStages(shaderStagesVector);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(1);
    rtPipelineInfo.setLayout(pipelineLayout.get());

    auto pipelineResult = context.getDevice().createRayTracingPipelineKHRUnique({}, {}, rtPipelineInfo);
    if (pipelineResult.result != vk::Result::eSuccess)
        throw std::runtime_error("failed to create ray tracing pipeline.");

    pipeline = std::move(pipelineResult.value);

    auto properties = context.getPhysicalDevice().getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.getDevice().getRayTracingShaderGroupHandlesKHR(pipeline.get(), 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess)
        throw std::runtime_error("failed to get ray tracing shader group handles.");

    uint32_t raygenSize = raygenCount * handleSizeAligned;
    uint32_t missSize = missCount * handleSizeAligned;
    uint32_t hitSize = hitCount * handleSizeAligned;

    raygenSBT = Buffer{context, Buffer::Type::ShaderBindingTable, raygenSize, handleStorage.data()};
    missSBT = Buffer{context, Buffer::Type::ShaderBindingTable, missSize, handleStorage.data() + raygenSize};
    hitSBT = Buffer{context, Buffer::Type::ShaderBindingTable, hitSize, handleStorage.data() + raygenSize + missSize};

    raygenRegion = vk::StridedDeviceAddressRegionKHR{raygenSBT.getDeviceAddress(), handleSizeAligned, raygenSize};
    missRegion = vk::StridedDeviceAddressRegionKHR{missSBT.getDeviceAddress(), handleSizeAligned, missSize};
    hitRegion = vk::StridedDeviceAddressRegionKHR{hitSBT.getDeviceAddress(), handleSizeAligned, hitSize};

    vk::DescriptorImageInfo storageImageInfo{};
    storageImageInfo.setImageView(outputImage.getImageView());
    storageImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet storageImageWrite{};
    storageImageWrite.setDstSet(descriptorSet.get());
    storageImageWrite.setDstBinding(1);
    storageImageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    storageImageWrite.setDescriptorCount(1);
    storageImageWrite.setImageInfo(storageImageInfo);

    context.getDevice().updateDescriptorSets(storageImageWrite, {});
}

GpuRaytracer::~GpuRaytracer()
{
    context.getDevice().waitIdle(); // Ensure GPU is idle before destroying resources.
    std::cout << "Destroying GpuRaytracer" << std::endl;
}

void GpuRaytracer::syncWithScene()
{
    context.getDevice().waitIdle();

    updateTLAS();
    updateMeshes();
    updateTextures();
}

void GpuRaytracer::updateTLAS()
{
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    
    {
        auto lock = scene.shared_lock();
        const auto& meshInstances = scene.getMeshInstances();
        instances.reserve(meshInstances.size());
        
        for (const auto* meshInstance : meshInstances)
            if (meshInstance)
                instances.push_back(meshInstance->getInstanceData());
    }

    if (instances.empty()) {
        tlas = Accel{};
        instancesBuffer = Buffer{};
    } else {
        instancesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR) * instances.size(), instances.data()};

        vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
        instancesData.setArrayOfPointers(false);
        instancesData.setData(instancesBuffer.getDeviceAddress());

        vk::AccelerationStructureGeometryKHR instanceGeometry;
        instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
        instanceGeometry.setGeometry({instancesData});
        instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        tlas.build(context, instanceGeometry, static_cast<uint32_t>(instances.size()), vk::AccelerationStructureTypeKHR::eTopLevel);
    }

    vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
    accelInfo.setAccelerationStructureCount(1);
    accelInfo.setPAccelerationStructures(&tlas.getAccelerationStructure());

    vk::WriteDescriptorSet accelWrite{};
    accelWrite.setDstSet(descriptorSet.get());
    accelWrite.setDstBinding(0);
    accelWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
    accelWrite.setDescriptorCount(1);
    accelWrite.setPNext(&accelInfo);

    context.getDevice().updateDescriptorSets(accelWrite, {});
}

void GpuRaytracer::updateMeshes()
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

void GpuRaytracer::updateTextures()
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

void GpuRaytracer::render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR, 0, sizeof(PushConstants), &pushConstants);
    commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, width, height, 1);
}
