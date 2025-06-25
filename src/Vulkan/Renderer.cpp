#include <memory>
#include "Renderer.h"

#include <iostream>
#include <ranges>

#include "Buffer.h"
#include "Globals.h"
#include "Utils.h"

Renderer::Renderer(Context& context, uint32_t width, uint32_t height)
: inputImage(context, width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst), dirty(false), width(width), height(height), context(context)
{
    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(context.surface.get());
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({width, height});
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
        #embed "../shaders/PathTracing/RayGeneration.spv"
    };
    static constexpr unsigned char PathTracingMiss[] = {
        #embed "../shaders/PathTracing/Miss.spv"
    };
    static constexpr unsigned char PathTracingClosestHit[] = {
        #embed "../shaders/PathTracing/ClosestHit.spv"
    };

    // Shader bytecode array
    constexpr const unsigned char* shaders[] = {
        RayGeneration,
        PathTracingMiss,
        PathTracingClosestHit,
    };

    // Shader sizes
    constexpr size_t shaderSizes[] = {
        sizeof(RayGeneration),
        sizeof(PathTracingMiss),
        sizeof(PathTracingClosestHit),
    };

    // Shader stages
    constexpr vk::ShaderStageFlagBits shaderStages[] = {
        vk::ShaderStageFlagBits::eRaygenKHR,
        vk::ShaderStageFlagBits::eMissKHR,
        vk::ShaderStageFlagBits::eClosestHitKHR,
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
        {2, vk::DescriptorType::eStorageBuffer, 1,  vk::ShaderStageFlagBits::eClosestHitKHR},
        {3, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR}
    };

    // Descriptor Binding Flags for Bindless
    std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(), vk::DescriptorBindingFlags{});
    bindingFlags[3] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.setBindingFlags(bindingFlags);

    // Create Descriptor Set Layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
    descSetLayoutInfo.setBindings(bindings);
    descSetLayoutInfo.setPNext(&bindingFlagsInfo);
    descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    // Allocate Descriptor Set
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountAllocInfo{};
    variableCountAllocInfo.setDescriptorCounts(MAX_TEXTURES);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.descPool.get());
    allocInfo.setSetLayouts(descSetLayout.get());
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPNext(&variableCountAllocInfo);

    descriptorSet = std::move(context.device->allocateDescriptorSetsUnique(allocInfo).front());

    // Create Pipeline Layout
    vk::PushConstantRange pushRange{};
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstants));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(descSetLayout.get());
    pipelineLayoutInfo.setPushConstantRanges(pushRange);

    pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    // Create Ray Tracing Pipeline
    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo{};
    rtPipelineInfo.setStages(shaderStagesVector);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(1);
    rtPipelineInfo.setLayout(pipelineLayout.get());

    auto pipelineResult = context.device->createRayTracingPipelineKHRUnique({}, {}, rtPipelineInfo);
    if (pipelineResult.result != vk::Result::eSuccess)
        throw std::runtime_error("failed to create ray tracing pipeline.");

    pipeline = std::move(pipelineResult.value);

    // Ray Tracing Properties
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    // Get Shader Group Handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(pipeline.get(), 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess)
        throw std::runtime_error("failed to get ray tracing shader group handles.");

    // Calculate SBT Region Sizes
    uint32_t raygenSize = raygenCount * handleSizeAligned;
    uint32_t missSize = missCount * handleSizeAligned;
    uint32_t hitSize  = hitCount * handleSizeAligned;

    // Create Shader Binding Table Buffers
    raygenSBT = Buffer{context, Buffer::Type::ShaderBindingTable, raygenSize, handleStorage.data()};
    missSBT   = Buffer{context, Buffer::Type::ShaderBindingTable, missSize, handleStorage.data() + raygenSize};
    hitSBT    = Buffer{context, Buffer::Type::ShaderBindingTable, hitSize, handleStorage.data() + raygenSize + missSize};

    // Create Strided Device Address Regions
    raygenRegion = vk::StridedDeviceAddressRegionKHR{ raygenSBT.getDeviceAddress(), handleSizeAligned, raygenSize };
    missRegion   = vk::StridedDeviceAddressRegionKHR{ missSBT.getDeviceAddress(), handleSizeAligned, missSize };
    hitRegion    = vk::StridedDeviceAddressRegionKHR{ hitSBT.getDeviceAddress(), handleSizeAligned, hitSize };

    // Bind Storage Image to Descriptor Set
    vk::DescriptorImageInfo storageImageInfo{};
    storageImageInfo.setImageView(inputImage.view.get());
    storageImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet storageImageWrite{};
    storageImageWrite.setDstSet(descriptorSet.get());
    storageImageWrite.setDstBinding(1);
    storageImageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
    storageImageWrite.setDescriptorCount(1);
    storageImageWrite.setImageInfo(storageImageInfo);

    context.device->updateDescriptorSets(storageImageWrite, {});
}

void Renderer::rebuildTLAS() {
    
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    instances.reserve(sceneObjects.size());

    for (const auto& objPtr : sceneObjects)
        if (auto* meshInstance = dynamic_cast<MeshInstance*>(objPtr.get()))
            instances.push_back(meshInstance->instanceData);  //always latest data
    
    instancesBuffer = Buffer{context,Buffer::Type::AccelInput,sizeof(vk::AccelerationStructureInstanceKHR) * instances.size(),instances.data()};

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
    accelInfo.setPAccelerationStructures(&tlas.accel.get());

    vk::WriteDescriptorSet accelWrite{};
    accelWrite.setDstSet(descriptorSet.get());
    accelWrite.setDstBinding(0);
    accelWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
    accelWrite.setDescriptorCount(1);
    accelWrite.setPNext(&accelInfo);

    context.device->updateDescriptorSets(accelWrite, {});
    markDirty();
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
    write.setDstBinding(3);
    write.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
    write.setDescriptorCount(static_cast<uint32_t>(textureImageInfos.size()));
    write.setImageInfo(textureImageInfos);

    context.device->updateDescriptorSets(write, {});
}

void Renderer::render(const uint32_t imageIndex, const PushConstants& pushConstants)
{
    const vk::CommandBuffer commandBuffer = getCommandBuffer(imageIndex);
    commandBuffer.begin(vk::CommandBufferBeginInfo());
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline.get());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipelineLayout.get(), 0, descriptorSet.get(), {});
    commandBuffer.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR, 0, sizeof(PushConstants), &pushConstants);
    commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, width, height, 1);
}

const vk::CommandBuffer& Renderer::getCommandBuffer(const uint32_t imageIndex) const
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
    textureNames.push_back(textures.back().getName());
}

std::shared_ptr<MeshAsset> Renderer::get(const std::string& name) const
{
    for (const auto& meshAsset : meshAssets)
        if (meshAsset->path == name)
            return meshAsset;
    return nullptr;
}

int Renderer::add(std::unique_ptr<SceneObject> sceneObject) {
    // If the object is a PerspectiveCamera, save a raw pointer to activeCamera
    if (auto camera = dynamic_cast<PerspectiveCamera*>(sceneObject.get()))
        activeCamera = camera;

    sceneObjects.push_back(std::move(sceneObject));
    rebuildTLAS();

    return sceneObjects.size() - 1;
}

void Renderer::rebuildMeshBuffer() {
    std::vector<MeshAddresses> meshAddresses;
    meshAddresses.reserve(meshAssets.size());
    for (const auto& meshAsset : meshAssets)
        meshAddresses.push_back(meshAsset->getBufferAddresses());
    updateStorageBuffer(2, meshAddresses, meshBuffer);
    markDirty();
}

void Renderer::add(const std::shared_ptr<MeshAsset>& meshAsset) {
    meshAsset->setMeshIndex(static_cast<uint32_t>(meshAssets.size()));
    meshAssets.push_back(meshAsset);
    rebuildMeshBuffer();
}

bool Renderer::remove(const SceneObject* obj) {
    // Don't remove the active camera
    if (activeCamera == obj) {
        std::cerr << "Warning: Cannot remove active camera from scene." << std::endl;
        return false;
    }

    const auto it = std::ranges::find_if(sceneObjects,
        [obj](const std::unique_ptr<SceneObject>& ptr) {
            return ptr.get() == obj;
        });

    if (it != sceneObjects.end())
    {
        sceneObjects.erase(it);
        rebuildTLAS();
        return true;
    }
    return false;
}
