#include "RtxRaytracer.h"

#include <iostream>
#include <ranges>
#include <algorithm>

#include "Globals.h"
#include "Scene/MeshInstance.h"

RtxRaytracer::RtxRaytracer(Scene& scene, uint32_t width, uint32_t height) : GpuRaytracer(scene, width, height)
{
    // Embed shader binaries
    static constexpr unsigned char RayGeneration[] = {
        #embed "../Shaders/RTX/RayGeneration.spv"
    };
    static constexpr unsigned char PathTracingMiss[] = {
        #embed "../Shaders/RTX/Miss.spv"
    };
    static constexpr unsigned char PathTracingClosestHit[] = {
        #embed "../Shaders/RTX/ClosestHit.spv"
    };
    static constexpr unsigned char VolumeIntersection[] = {
        #embed "../Shaders/RTX/IntersectionVolume.spv"
    };
    static constexpr unsigned char VolumeClosestHit[] = {
        #embed "../Shaders/RTX/ClosestHitVolume.spv"
    };
    
    constexpr const unsigned char* shaders[] = {
        RayGeneration,       // Index 0
        PathTracingMiss,       // Index 1
        PathTracingClosestHit, // Index 2
        VolumeIntersection,    // Index 3
        VolumeClosestHit       // Index 4
    };

    // Correctly get the size of each embedded shader binary.
    constexpr size_t shaderSizes[] = {
        sizeof(RayGeneration),
        sizeof(PathTracingMiss),
        sizeof(PathTracingClosestHit),
        sizeof(VolumeIntersection),
        sizeof(VolumeClosestHit)
    };
    
    constexpr vk::ShaderStageFlagBits shaderStages[] = {
        vk::ShaderStageFlagBits::eRaygenKHR,
        vk::ShaderStageFlagBits::eMissKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR,
        vk::ShaderStageFlagBits::eIntersectionKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR
    };

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesVector;
    shaderModules.reserve(std::size(shaders));
    shaderStagesVector.reserve(std::size(shaders));

    // Create shader modules and pipeline stage info for all shaders.
    for (size_t i = 0; i < std::size(shaders); ++i)
    {
        shaderModules.emplace_back(context.getDevice().createShaderModuleUnique({{}, shaderSizes[i], reinterpret_cast<const uint32_t*>(shaders[i])}));
        shaderStagesVector.push_back({{}, shaderStages[i], *shaderModules.back(), "main"});
    }

    // . The order here determines the order in the Shader Binding Table (SBT) and the offset in the instances
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;
    // Group 0: Ray Generation
    shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    // Group 1: Miss Shader
    shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    // Group 2: Triangle Hit Group (for standard mesh geometry) SBT offset 0.
    shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    // Group 3: Volume Procedural Hit Group (for volume geometry) SBT offset of 1.
    shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup, VK_SHADER_UNUSED_KHR, 4, VK_SHADER_UNUSED_KHR, 3);
    
    // Define the counts for SBT calculation.
    uint32_t raygenCount = 1;
    uint32_t missCount = 1;
    uint32_t hitCount = 2; // One for triangles, one for volumes.

    // DESCRIPTOR SET AND PIPELINE LAYOUT
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Output emission image
        {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Output albedo image
        {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Output normal image
        {4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Output crypto image
        {5, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Output position image
        {6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR}, // Mesh instances buffer
        {7, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR}, // Textures
    };

    createDescriptorSet(bindings);

    vk::PushConstantRange pushRange{};
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstantsData));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(descSetLayout.get());
    pipelineLayoutInfo.setPushConstantRanges(pushRange);

    pipelineLayout = context.getDevice().createPipelineLayoutUnique(pipelineLayoutInfo);

    // --- RAY TRACING PIPELINE AND SBT CREATION ---

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
    
    bindOutputImages();
}


void RtxRaytracer::updateTLAS()
{
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    const auto& meshInstances = scene.getMeshInstances();
    instances.reserve(meshInstances.size());
    
    for (const auto* meshInstance : meshInstances)
        if (meshInstance)
            instances.push_back(meshInstance->getInstanceData());

    if (instances.empty())
    {
        // Create a dummy instance if the scene is empty to avoid creating an empty buffer
        auto emptyInstance = vk::AccelerationStructureInstanceKHR{};
        instancesBuffer = Buffer(context, Buffer::Type::AccelInput,sizeof(vk::AccelerationStructureInstanceKHR),&emptyInstance);
    }
    else
    {
        instancesBuffer = Buffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR) * instances.size(), instances.data()};
    }
    
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.getDeviceAddress());

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    tlas.build(context, instanceGeometry, static_cast<uint32_t>(instances.size()), vk::AccelerationStructureTypeKHR::eTopLevel);

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

void RtxRaytracer::render(const vk::CommandBuffer& commandBuffer, const PushConstantsData& pushConstants)
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR, 0, sizeof(PushConstantsData), &pushConstants);
    commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, width, height, 1);
}