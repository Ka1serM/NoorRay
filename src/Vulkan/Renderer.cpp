#include <memory>
#include "Renderer.h"
#include "Buffer.h"
#include "Globals.h"
#include "Utils.h"

Renderer::Renderer(Context& context) : dirty(false), context(context) {

    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.surface.get());
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({WIDTH, HEIGHT});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst |vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eMailbox);
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context.queueFamilyIndex);

    swapchain = context.device->createSwapchainKHRUnique(swapchainInfo);
    swapchainImages = context.device->getSwapchainImagesKHR(swapchain.get());

    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(*context.commandPool);
    commandBufferInfo.setCommandBufferCount(static_cast<uint32_t>(swapchainImages.size()));
    commandBuffers = context.device->allocateCommandBuffersUnique(commandBufferInfo);

    // Embed all shaders as constexpr unsigned char arrays
    static constexpr unsigned char RayGeneration[] = {
        #embed "../shaders/RayGeneration.spv"
    };
    static constexpr unsigned char RayTracingMiss[] = {
        #embed "../shaders/RayTracing/Miss.spv"
    };
    static constexpr unsigned char PathTracingMiss[] = {
        #embed "../shaders/PathTracing/Miss.spv"
    };
    static constexpr unsigned char ShadowRayMiss[] = {
        #embed "../shaders/ShadowRay/Miss.spv"
    };
    static constexpr unsigned char RayTracingClosestHit[] = {
        #embed "../shaders/RayTracing/ClosestHit.spv"
    };
    static constexpr unsigned char PathTracingClosestHit[] = {
        #embed "../shaders/PathTracing/ClosestHit.spv"
    };
    static constexpr unsigned char ShadowRayClosestHit[] = {
        #embed "../shaders/ShadowRay/ClosestHit.spv"
    };

    // Parallel arrays for shaders, sizes, and stages
    constexpr const unsigned char* shaders[] = {
        RayGeneration,
        RayTracingMiss,
        PathTracingMiss,
        ShadowRayMiss,
        RayTracingClosestHit,
        PathTracingClosestHit,
        ShadowRayClosestHit
    };

    constexpr size_t shaderSizes[] = {
        sizeof(RayGeneration),
        sizeof(RayTracingMiss),
        sizeof(PathTracingMiss),
        sizeof(ShadowRayMiss),
        sizeof(RayTracingClosestHit),
        sizeof(PathTracingClosestHit),
        sizeof(ShadowRayClosestHit)
    };

    constexpr vk::ShaderStageFlagBits shaderStages[] = {
        vk::ShaderStageFlagBits::eRaygenKHR,
        vk::ShaderStageFlagBits::eMissKHR,
        vk::ShaderStageFlagBits::eMissKHR,
        vk::ShaderStageFlagBits::eMissKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR
    };

    // Create shader modules, stages, and shader groups
    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesVector;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    uint32_t raygenCount = 0;
    uint32_t missCount = 0;
    uint32_t hitCount = 0;

    for (size_t i = 0; i < std::size(shaders); ++i) {
        shaderModules.emplace_back(context.device->createShaderModuleUnique({
            {}, shaderSizes[i], reinterpret_cast<const uint32_t*>(shaders[i])
        }));

        shaderStagesVector.push_back({{}, shaderStages[i], *shaderModules.back(), "main"});

        if (shaderStages[i] == vk::ShaderStageFlagBits::eRaygenKHR) {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            raygenCount++;
        } else if (shaderStages[i] == vk::ShaderStageFlagBits::eMissKHR) {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            missCount++;
        } else if (shaderStages[i] == vk::ShaderStageFlagBits::eClosestHitKHR) {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            hitCount++;
        }
    }

    // Descriptor Set Layout Bindings
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR},
        {5, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR}
    };

    //Descriptor binding flags for Bindless
    std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(), vk::DescriptorBindingFlags{});
    bindingFlags[5] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
    //AllocationInfo for Bindless
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountAllocInfo{};
    variableCountAllocInfo.setDescriptorCounts(MAX_TEXTURES);

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.setBindingFlags(bindingFlags);

    //Create descriptor set layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
    descSetLayoutInfo.setBindings(bindings);
    descSetLayoutInfo.setPNext(&bindingFlagsInfo);

    descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.descPool.get());
    allocInfo.setSetLayouts(descSetLayout.get());
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPNext(&variableCountAllocInfo);

    descriptorSet = std::move(context.device->allocateDescriptorSetsUnique(allocInfo).front());

    //Create pipeline layout
    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstants));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayout.get());
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    //Create pipeline
    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo;
    rtPipelineInfo.setStages(shaderStagesVector);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(1);
    rtPipelineInfo.setLayout(pipelineLayout.get());

    auto pipelineResult = context.device->createRayTracingPipelineKHRUnique({}, {}, rtPipelineInfo);
    if (pipelineResult.result != vk::Result::eSuccess)
        throw std::runtime_error("failed to create ray tracing pipeline.");

    pipeline = std::move(pipelineResult.value);

    //Get ray tracing properties
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    //Calculate shader binding table (SBT) size
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = shaderGroups.size();
    uint32_t sbtSize = groupCount * handleSizeAligned;

    //Get shader group handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(pipeline.get(), 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess)
        throw std::runtime_error("failed to get ray tracing shader group handles.");

    //Create Shader Binding Table (SBT)
    //Calculate the total size for each group
    uint32_t raygenSize = raygenCount * handleSizeAligned;
    uint32_t missSize = missCount * handleSizeAligned;
    uint32_t hitSize = hitCount * handleSizeAligned;

    //Create SBT buffers
    raygenSBT = Buffer{context,Buffer::Type::ShaderBindingTable,raygenSize,handleStorage.data()};
    missSBT = Buffer{context,Buffer::Type::ShaderBindingTable,missSize,handleStorage.data() + raygenSize};
    hitSBT= Buffer{context,Buffer::Type::ShaderBindingTable,hitSize,handleStorage.data() + (raygenSize + hitSize)};

    //Create StridedDeviceAddressRegion
    raygenRegion = vk::StridedDeviceAddressRegionKHR{ raygenSBT.getDeviceAddress(), handleSizeAligned, raygenSize};
    missRegion = vk::StridedDeviceAddressRegionKHR{ missSBT.getDeviceAddress(), handleSizeAligned, missSize };
    hitRegion = vk::StridedDeviceAddressRegionKHR{ hitSBT.getDeviceAddress(), handleSizeAligned, hitSize };
}

void Renderer::buildTLAS() {
    std::vector<vk::AccelerationStructureInstanceKHR> instanceData;
    instanceData.reserve(instancePtrs.size());

    for (const auto* instPtr : instancePtrs)
        instanceData.push_back(*instPtr); // fresh copy of the latest transform and data

    instancesBuffer = Buffer{context,Buffer::Type::AccelInput,sizeof(vk::AccelerationStructureInstanceKHR) * instanceData.size(),instanceData.data()};

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.getDeviceAddress());

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    tlas.build(context, instanceGeometry, static_cast<uint32_t>(instanceData.size()), vk::AccelerationStructureTypeKHR::eTopLevel);

    vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
    accelInfo.setAccelerationStructureCount(1);
    accelInfo.setPAccelerationStructures(&tlas.accel.get());

    vk::WriteDescriptorSet accelWrite{};
    accelWrite.setDstSet(descriptorSet.get());
    accelWrite.setDstBinding(0);
    accelWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
    accelWrite.setDescriptorCount(1);
    accelWrite.setPNext(&accelInfo);

    context.device->updateDescriptorSets(accelWrite, {});
}

void Renderer::updateStorageImage(const vk::ImageView& storageImageView) {
    vk::DescriptorImageInfo storageImageInfo{};
    storageImageInfo.setImageView(storageImageView);
    storageImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet storageImageWrite{};
    storageImageWrite.setDstSet(descriptorSet.get());
    storageImageWrite.setDstBinding(1);
    storageImageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    storageImageWrite.setDescriptorCount(1);
    storageImageWrite.setImageInfo(storageImageInfo);

    context.device->updateDescriptorSets(storageImageWrite, {});
}

template<typename T>
void Renderer::updateStorageBuffer(uint32_t binding, const std::vector<T>& data, Buffer& buffer) {
    buffer = Buffer{context, Buffer::Type::Storage, sizeof(T) * data.size(), data.data()};

    vk::WriteDescriptorSet write{};
    write.setDstSet(descriptorSet.get());
    write.setDstBinding(binding);
    write.setDescriptorType(vk::DescriptorType::eStorageBuffer);
    write.setDescriptorCount(1);
    write.setBufferInfo(buffer.getDescriptorInfo());

    context.device->updateDescriptorSets(write, {});
}

void Renderer::updateTextureDescriptors(const std::vector<Texture>& textures) {
    textureImageInfos.clear();
    for (const auto& texture : textures)
        textureImageInfos.push_back(texture.getDescriptorInfo());

    vk::WriteDescriptorSet write{};
    write.setDstSet(descriptorSet.get());
    write.setDstBinding(5);
    write.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
    write.setDescriptorCount(static_cast<uint32_t>(textureImageInfos.size()));
    write.setImageInfo(textureImageInfos);

    context.device->updateDescriptorSets(write, {});
}

void Renderer::render(uint32_t imageIndex, const PushConstants& pushConstants)
{
    const vk::CommandBuffer commandBuffer = getCommandBuffer(imageIndex);
    commandBuffer.begin(vk::CommandBufferBeginInfo());
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR, 0, sizeof(PushConstants), &pushConstants);
    commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);
}

const vk::CommandBuffer& Renderer::getCommandBuffer(uint32_t imageIndex) const
{
    return commandBuffers[imageIndex].get();
}

const vk::SwapchainKHR& Renderer::getSwapChain() const
{
    return swapchain.get();
}

const std::vector<vk::Image>& Renderer::getSwapchainImages() const
{
    return swapchainImages;
}

void Renderer::add(Texture&& element) {
    textures.push_back(std::move(element)); // move only
    updateTextureDescriptors(textures);
}

void Renderer::add(const PointLight& element) {
    pointLights.push_back(element);
    updateStorageBuffer(4, pointLights, pointLightBuffer);
}

void Renderer::add(const MeshAddresses& element) {
    meshAddresses.push_back(element);
    updateStorageBuffer(2, meshAddresses, meshBuffer);
}

void Renderer::add(const Material& element) {
    materials.push_back(element);
    updateStorageBuffer(3, materials, materialBuffer);
}

void Renderer::add(const vk::AccelerationStructureInstanceKHR& element) {
    instancePtrs.push_back(&element);
    buildTLAS();
}

// Vector add overloads
void Renderer::add(std::vector<Texture>&& elements) {
    textures.insert(textures.end(), std::make_move_iterator(elements.begin()), std::make_move_iterator(elements.end()));
    updateTextureDescriptors(textures);
}

void Renderer::add(const std::vector<PointLight>& elements) {
    pointLights.insert(pointLights.end(), elements.begin(), elements.end());
    updateStorageBuffer(4, pointLights, pointLightBuffer);
}

void Renderer::add(const std::vector<MeshAddresses>& elements) {
    meshAddresses.insert(meshAddresses.end(), elements.begin(), elements.end());
    updateStorageBuffer(2, meshAddresses, meshBuffer);
}

void Renderer::add(const std::vector<Material>& elements) {
    materials.insert(materials.end(), elements.begin(), elements.end());
    updateStorageBuffer(3, materials, materialBuffer);
}

void Renderer::add(const std::vector<const vk::AccelerationStructureInstanceKHR*>& elements) {
    instancePtrs.insert(instancePtrs.end(), elements.begin(), elements.end());
    buildTLAS();
}