#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>

#include "Context.h"
#include "Buffer.h"
#include "Image.h"
#include "Accel.h"
#include "Utils.h"
#include "Camera.h"
#include "ConvertImageComputeShader.h"
#include "Globals.h"
#include "Texture.h"

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

    Image inputImage{context, WIDTH, HEIGHT, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};
    Image outputImage{context,WIDTH, HEIGHT, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};

    //REGION BUFFERS
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    std::vector<Material> materials;
    std::vector<PointLight> pointLights;

    // PointLight placeholder{};
    // placeholder.intensity = 1.0f;
    // placeholder.color = glm::vec3(1.0f, 1.0f, 1.0f);
    // placeholder.position = glm::vec3(1000.0f, -1000.0f, 0.0f);
    // pointLights.push_back(placeholder);

    std::vector<Texture> textures;
    Utils::loadObj(context, "../assets/LittleTokyo.obj", vertices, indices, faces, materials, pointLights, textures);
    //Utils::loadGltf("../assets/Valorant.gltf", vertices, indices, faces, materials, pointLights);

    Buffer vertexBuffer{context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data()};
    Buffer indexBuffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    Buffer faceBuffer{context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};
    Buffer materialBuffer{context, Buffer::Type::Storage, sizeof(Material) * materials.size(), materials.data()};
    Buffer pointLightBuffer{context, Buffer::Type::Storage, sizeof(PointLight) * pointLights.size(), pointLights.data()};

    std::vector<vk::DescriptorImageInfo> textureImageInfos;
    for (const auto& texture : textures)
        textureImageInfos.push_back(texture.getDescriptorInfo());
    Buffer textureBuffer{context, Buffer::Type::Storage, sizeof(vk::DescriptorImageInfo) * textureImageInfos.size(), textureImageInfos.data()};

    //Create bottom level accel structure
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

    //Create top level accel structure
    std::vector<vk::AccelerationStructureInstanceKHR> accelInstances;

    int gridSize = 4;
    float spacing = 100.0f;

    for (int x = 0; x < gridSize; ++x) {
        for (int z = 0; z < gridSize; ++z) {
            vk::TransformMatrixKHR transformMatrix = std::array{
                std::array{1.0f, 0.0f, 0.0f, x * spacing},
                std::array{0.0f, 1.0f, 0.0f, 0.0f},
                std::array{0.0f, 0.0f, 1.0f, z * spacing},
            };

            vk::AccelerationStructureInstanceKHR accelInstance;
            accelInstance.setTransform(transformMatrix);
            accelInstance.setMask(0xFF);
            accelInstance.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress);
            accelInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

            accelInstances.push_back(accelInstance);
        }
    }

    Buffer instancesBuffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR) * accelInstances.size(), accelInstances.data()};

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    Accel topAccel{context, instanceGeometry, static_cast<uint32_t>(accelInstances.size()), vk::AccelerationStructureTypeKHR::eTopLevel};

    std::vector<std::pair<std::string, vk::ShaderStageFlagBits>> shaderPaths = {
        {"../src/shaders/RayGeneration.spv", vk::ShaderStageFlagBits::eRaygenKHR},
        {"../src/shaders/RayTracing/Miss.spv", vk::ShaderStageFlagBits::eMissKHR},
        {"../src/shaders/PathTracing/Miss.spv", vk::ShaderStageFlagBits::eMissKHR},
        {"../src/shaders/ShadowRay/Miss.spv", vk::ShaderStageFlagBits::eMissKHR},
        {"../src/shaders/RayTracing/ClosestHit.spv", vk::ShaderStageFlagBits::eClosestHitKHR},
        {"../src/shaders/PathTracing/ClosestHit.spv", vk::ShaderStageFlagBits::eClosestHitKHR},
        {"../src/shaders/ShadowRay/ClosestHit.spv", vk::ShaderStageFlagBits::eClosestHitKHR}
    };

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    uint32_t raygenCount = 0;
    uint32_t missCount = 0;
    uint32_t hitCount = 0;
    for (uint32_t i = 0; i < shaderPaths.size(); ++i) {
        const auto& [path, stage] = shaderPaths[i];
        auto code = Utils::readFile(path);
        shaderModules.push_back(context.device->createShaderModuleUnique({{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())}));
        shaderStages.push_back({{}, stage, *shaderModules.back(), "main"});

        if (stage == vk::ShaderStageFlagBits::eRaygenKHR) {
            shaderGroups.push_back({vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR});
            raygenCount++;
        } else if (stage == vk::ShaderStageFlagBits::eMissKHR) {
            shaderGroups.push_back({vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR});
            missCount++;
        } else if (stage == vk::ShaderStageFlagBits::eClosestHitKHR) {
            shaderGroups.push_back({vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR});
            hitCount++;
        }
    }

    // Descriptor Set Layout Bindings
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},
        {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR},
        {6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR},
        {7, vk::DescriptorType::eCombinedImageSampler, MAX_TEXTURES, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR}
    };

    //Descriptor binding flags for bindless
    std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(), vk::DescriptorBindingFlags{});
    bindingFlags[7] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.setBindingFlags(bindingFlags);

    //Create descriptor set layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
    descSetLayoutInfo.setBindings(bindings);
    descSetLayoutInfo.setPNext(&bindingFlagsInfo);

    vk::UniqueDescriptorSetLayout descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    //Descriptor set allocation with variable descriptor count
    uint32_t actualTextureCount = static_cast<uint32_t>(textureImageInfos.size());

    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountAllocInfo{};
    variableCountAllocInfo.setDescriptorCounts(actualTextureCount);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context.descPool.get());
    allocInfo.setSetLayouts(*descSetLayout);
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPNext(&variableCountAllocInfo);

    std::vector<vk::UniqueDescriptorSet> descSets = context.device->allocateDescriptorSetsUnique(allocInfo);
    vk::UniqueDescriptorSet descSet = std::move(descSets[0]);

    // Descriptor writes
    std::vector<vk::WriteDescriptorSet> writes(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        writes[i].setDstSet(*descSet);
        writes[i].setDstBinding(bindings[i].binding);
        writes[i].setDescriptorType(bindings[i].descriptorType);
        writes[i].setDescriptorCount(bindings[i].descriptorCount);
    }

    // Now assign the descriptor-specific info
    writes[0].setPNext(&topAccel.descAccelInfo);
    writes[1].setImageInfo(inputImage.descImageInfo);
    writes[2].setBufferInfo(vertexBuffer.descBufferInfo);
    writes[3].setBufferInfo(indexBuffer.descBufferInfo);
    writes[4].setBufferInfo(faceBuffer.descBufferInfo);
    writes[5].setBufferInfo(materialBuffer.descBufferInfo);
    writes[6].setBufferInfo(pointLightBuffer.descBufferInfo);
    writes[7].setImageInfo(textureImageInfos);
    writes[7].setDescriptorCount(actualTextureCount);

    //Final descriptor update
    context.device->updateDescriptorSets(writes, {});

    //Create pipeline layout
    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstants));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    vk::UniquePipelineLayout pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    //Create pipeline
    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo;
    rtPipelineInfo.setStages(shaderStages);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(1);
    rtPipelineInfo.setLayout(*pipelineLayout);

    auto result = context.device->createRayTracingPipelineKHRUnique({}, {}, rtPipelineInfo);
    if (result.result != vk::Result::eSuccess)
        throw std::runtime_error("failed to create ray tracing pipeline.");

    vk::UniquePipeline pipeline = std::move(result.value);

    //Get ray tracing properties
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    //Calculate shader binding table (SBT) size
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    //Get shader group handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess)
        throw std::runtime_error("failed to get ray tracing shader group handles.");

    //Create Shader Binding Table (SBT)
    //Calculate the total size for each group
    uint32_t raygenSize = raygenCount * handleSizeAligned;
    uint32_t missSize = missCount * handleSizeAligned;
    uint32_t hitSize = hitCount * handleSizeAligned;

    //Create SBT buffers
    Buffer raygenSBT{context,Buffer::Type::ShaderBindingTable,raygenSize,handleStorage.data()};
    Buffer missSBT{context,Buffer::Type::ShaderBindingTable,missSize,handleStorage.data() + raygenSize};
    Buffer hitSBT{context,Buffer::Type::ShaderBindingTable,hitSize,handleStorage.data() + (raygenSize + hitSize)};

    //Create StridedDeviceAddressRegion
    vk::StridedDeviceAddressRegionKHR raygenRegion{ raygenSBT.deviceAddress, handleSizeAligned, raygenSize};
    vk::StridedDeviceAddressRegionKHR missRegion{ missSBT.deviceAddress, handleSizeAligned, missSize };
    vk::StridedDeviceAddressRegionKHR hitRegion{ hitSBT.deviceAddress, handleSizeAligned, hitSize };

    //CAMERA
    static PerspectiveCamera camera(glm::vec3(0, -1, 3.25), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), static_cast<float>(WIDTH)/HEIGHT, 105);

    //Setup Input
    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); //capture cursor

    static bool isRayTracing = false;
    glfwSetKeyCallback(context.window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_P && action == GLFW_PRESS) {
            isRayTracing = !isRayTracing;
            std::cout << (isRayTracing ? "Switched to simple Ray Tracing" : "Switched to Path Tracing") << std::endl;
        }
    });

    InputTracker inputTracker = InputTracker(context.window);

    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());

    ConvertImageComputeShader convertImage("../src/shaders/ConvertImage.spv", context, inputImage.view.get(), outputImage.view.get());

    //Main loop
    int frame = 0;
    uint32_t imageIndex = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;

    while (!glfwWindowShouldClose(context.window)) {
        //Delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        timeAccumulator += deltaTime;
        frameCounter++;

        if (timeAccumulator >= 1.0f) {
            std::cout << "Average FPS: " << frameCounter / timeAccumulator << std::endl;
            timeAccumulator = 0.0f;
            frameCounter = 0;
        }

        //Camera Update
        glfwPollEvents();
        inputTracker.update();

        //Push Constants
        PushConstants pushConstantData;
        pushConstantData.push.frame = frame;
        pushConstantData.push.isRayTracing = isRayTracing;
        pushConstantData.camera = camera.update(inputTracker, deltaTime);

        //Acquire next image
        imageIndex = context.device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore).value;

        //Record commands
        vk::CommandBuffer commandBuffer = *commandBuffers[imageIndex];
        commandBuffer.begin(vk::CommandBufferBeginInfo());
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, {});
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR, 0, sizeof(PushConstants), &pushConstantData);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        //Convert 32bit float to 8bit image for display, read from inputImage and write to outputImage
        convertImage.dispatch(commandBuffer, (WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);

        //Transition output image to presentable layout
        outputImage.transitionAndCopyTo(commandBuffer, swapchainImages[imageIndex], {WIDTH, HEIGHT, 1});
        commandBuffer.end();

        //Submit
        context.queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        //Present image
        vk::PresentInfoKHR presentInfo;
        presentInfo.setSwapchains(*swapchain);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);
        auto result = context.queue.presentKHR(presentInfo);

        if (result != vk::Result::eSuccess)
            throw std::runtime_error("failed to present.");

        context.queue.waitIdle();

        if (pushConstantData.camera.isMoving) //reset frame counter when cam is moving
            frame = 0;
        else
            frame++;
    }

    context.device->waitIdle();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}