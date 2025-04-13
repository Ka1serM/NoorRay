#include "Context.h"
#include "Buffer.h"
#include "Image.h"
#include "Accel.h"
#include "Mesh.h"
#include "Utils.h"

#include <vector>

constexpr int WIDTH = 800;
constexpr int HEIGHT = 800;

int main() {
    Context context(WIDTH, HEIGHT);

    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.setSurface(*context.surface);
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({WIDTH, HEIGHT});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context.queueFamilyIndex);
    vk::UniqueSwapchainKHR swapchain = context.device->createSwapchainKHRUnique(swapchainInfo);

    std::vector<vk::Image> swapchainImages = context.device->getSwapchainImagesKHR(*swapchain);

    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(*context.commandPool);
    commandBufferInfo.setCommandBufferCount(static_cast<uint32_t>(swapchainImages.size()));
    std::vector<vk::UniqueCommandBuffer> commandBuffers = context.device->allocateCommandBuffersUnique(commandBufferInfo);

    Image outputImage{context,
                      WIDTH, HEIGHT,
                      vk::Format::eB8G8R8A8Unorm,
                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};

    // Load mesh
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    Utils::loadFromFile(vertices, indices, faces);

    Buffer vertexBuffer{context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data()};
    Buffer indexBuffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    Buffer faceBuffer{context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};

    // Create bottom level accel struct
    vk::AccelerationStructureGeometryTrianglesDataKHR triangleData;
    triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triangleData.setVertexData(vertexBuffer.deviceAddress);
    triangleData.setVertexStride(sizeof(Vertex));
    triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
    triangleData.setIndexType(vk::IndexType::eUint32);
    triangleData.setIndexData(indexBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR triangleGeometry;
    triangleGeometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    triangleGeometry.setGeometry({triangleData});
    triangleGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    const auto primitiveCount = static_cast<uint32_t>(indices.size() / 3);

    Accel bottomAccel{context, triangleGeometry, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel};

    // Create top level accel struct
    vk::TransformMatrixKHR transformMatrix = std::array{
        std::array{1.0f, 0.0f, 0.0f, 0.0f},
        std::array{0.0f, 1.0f, 0.0f, 0.0f},
        std::array{0.0f, 0.0f, 1.0f, 0.0f},
    };

    vk::AccelerationStructureInstanceKHR accelInstance;
    accelInstance.setTransform(transformMatrix);
    accelInstance.setMask(0xFF);
    accelInstance.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress);
    accelInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    Buffer instancesBuffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR), &accelInstance};

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    Accel topAccel{context, instanceGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel};

    // Load shaders
    const std::vector<char> raygenCode = Utils::readFile("../shaders/raygen.rgen.spv");
    const std::vector<char> missCode = Utils::readFile("../shaders/miss.rmiss.spv");
    const std::vector<char> chitCode = Utils::readFile("../shaders/closesthit.rchit.spv");

    std::vector<vk::UniqueShaderModule> shaderModules(3);
    shaderModules[0] = context.device->createShaderModuleUnique({{}, raygenCode.size(), reinterpret_cast<const uint32_t*>(raygenCode.data())});
    shaderModules[1] = context.device->createShaderModuleUnique({{}, missCode.size(), reinterpret_cast<const uint32_t*>(missCode.data())});
    shaderModules[2] = context.device->createShaderModuleUnique({{}, chitCode.size(), reinterpret_cast<const uint32_t*>(chitCode.data())});

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages(3);
    shaderStages[0] = {{}, vk::ShaderStageFlagBits::eRaygenKHR, *shaderModules[0], "main"};
    shaderStages[1] = {{}, vk::ShaderStageFlagBits::eMissKHR, *shaderModules[1], "main"};
    shaderStages[2] = {{}, vk::ShaderStageFlagBits::eClosestHitKHR, *shaderModules[2], "main"};

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups(3);
    shaderGroups[0] = {vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    shaderGroups[1] = {vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    shaderGroups[2] = {vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

    // create ray tracing pipeline
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Binding = 0 : TLAS
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR}, // Binding = 1 : Storage image
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR}, // Binding = 2 : Vertices
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR}, // Binding = 3 : Indices
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR}, // Binding = 4 : Faces
    };

    // Create desc set layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo;
    descSetLayoutInfo.setBindings(bindings);
    vk::UniqueDescriptorSetLayout descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    // Create pipeline layout
    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(int));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    vk::UniquePipelineLayout pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    // Create pipeline
    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo;
    rtPipelineInfo.setStages(shaderStages);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(4);
    rtPipelineInfo.setLayout(*pipelineLayout);

    auto result = context.device->createRayTracingPipelineKHRUnique(nullptr, nullptr, rtPipelineInfo);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create ray tracing pipeline.");
    }

    vk::UniquePipeline pipeline = std::move(result.value);

    // Get ray tracing properties
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    // Calculate shader binding table (SBT) size
    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    // Get shader group handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
        throw std::runtime_error("failed to get ray tracing shader group handles.");
    }

    // Create SBT
    Buffer raygenSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned};
    Buffer missSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned};
    Buffer hitSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned};

    uint32_t stride = rtProperties.shaderGroupHandleAlignment;
    uint32_t size = rtProperties.shaderGroupHandleAlignment;

    vk::StridedDeviceAddressRegionKHR raygenRegion{raygenSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR missRegion{missSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR hitRegion{hitSBT.deviceAddress, stride, size};

    // Create desc set
    vk::UniqueDescriptorSet descSet = context.allocateDescSet(*descSetLayout);
    std::vector<vk::WriteDescriptorSet> writes(bindings.size());
    for (int i = 0; i < bindings.size(); i++) {
        writes[i].setDstSet(*descSet);
        writes[i].setDescriptorType(bindings[i].descriptorType);
        writes[i].setDescriptorCount(bindings[i].descriptorCount);
        writes[i].setDstBinding(bindings[i].binding);
    }
    writes[0].setPNext(&topAccel.descAccelInfo);
    writes[1].setImageInfo(outputImage.descImageInfo);
    writes[2].setBufferInfo(vertexBuffer.descBufferInfo);
    writes[3].setBufferInfo(indexBuffer.descBufferInfo);
    writes[4].setBufferInfo(faceBuffer.descBufferInfo);
    context.device->updateDescriptorSets(writes, nullptr);

    // Main loop
    uint32_t imageIndex = 0;
    int frame = 0;
    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    while (!glfwWindowShouldClose(context.window)) {
        glfwPollEvents();

        // Acquire next image
        imageIndex = context.device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore).value;

        // Record commands
        vk::CommandBuffer commandBuffer = *commandBuffers[imageIndex];
        commandBuffer.begin(vk::CommandBufferBeginInfo());
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(int), &frame);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        vk::Image srcImage = *outputImage.image;
        vk::Image dstImage = swapchainImages[imageIndex];
        Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
        Image::setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        Image::copyImage(commandBuffer, srcImage, dstImage, outputImage.info.extent);
        Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
        Image::setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

        commandBuffer.end();

        // Submit
        context.queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        // Present image
        vk::PresentInfoKHR presentInfo;
        presentInfo.setSwapchains(*swapchain);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);
        auto result = context.queue.presentKHR(presentInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present.");
        }
        context.queue.waitIdle();
        frame++;
    }

    context.device->waitIdle();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}