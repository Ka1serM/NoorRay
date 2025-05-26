#include <chrono>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <vector>

#include "Vulkan/Context.h"
#include "Vulkan/Buffer.h"
#include "Vulkan/Image.h"
#include "Vulkan/Accel.h"
#include "Utils.h"
#include "Camera/PerspectiveCamera.h"
#include "Vulkan/HdrToLdrCompute.h"
#include "Globals.h"
#include "UI/ImGuiManager.h"
#include "Mesh/MeshAsset.h"
#include "Scene/MeshInstance.h"
#include "UI/DebugPanel.h"

#include "UI/MainMenuBar.h"
#include "UI/OutlinerDetailsPanel.h"
#include "UI/ViewportPanel.h"

int main() {
    auto context = std::make_shared<Context>(WIDTH, HEIGHT);

    vk::SwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.setSurface(*context->surface);
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({WIDTH, HEIGHT});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst |vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eMailbox);
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context->queueFamilyIndex);

    vk::UniqueSwapchainKHR swapchain = context->device->createSwapchainKHRUnique(swapchainInfo);
    std::vector<vk::Image> swapchainImages = context->device->getSwapchainImagesKHR(*swapchain);

    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(*context->commandPool);
    commandBufferInfo.setCommandBufferCount(static_cast<uint32_t>(swapchainImages.size()));
    std::vector<vk::UniqueCommandBuffer> commandBuffers = context->device->allocateCommandBuffersUnique(commandBufferInfo);

    Image inputImage{context, WIDTH, HEIGHT, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};
    Image outputImage{context,WIDTH, HEIGHT, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst };

    auto scene = std::make_shared<Scene>(context);

    PointLight pointLight{};
    scene->pointLights.push_back(pointLight);

    auto meshAsset2 = std::make_shared<MeshAsset>(context, scene, "../assets/LittleTokyo.obj");
    scene->addMeshAsset(meshAsset2);

    auto meshInstance = std::make_unique<MeshInstance>(scene, "Mesh Instance 1", meshAsset2, Transform(glm::vec3(0, 100, 0), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1)));
    auto meshInstance2 = std::make_unique<MeshInstance>(scene, "Instance 2", meshAsset2, Transform());

    scene->addMeshInstance(std::move(meshInstance));
    scene->addMeshInstance(std::move(meshInstance2));

    scene->buildBuffersGPU();
    scene->buildTLAS();

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
        shaderModules.emplace_back(context->device->createShaderModuleUnique({{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())}));
        shaderStages.push_back({{}, stage, *shaderModules.back(), "main"});

        if (stage == vk::ShaderStageFlagBits::eRaygenKHR) {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            raygenCount++;
        } else if (stage == vk::ShaderStageFlagBits::eMissKHR) {
            shaderGroups.emplace_back(vk::RayTracingShaderGroupTypeKHR::eGeneral, i, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
            missCount++;
        } else if (stage == vk::ShaderStageFlagBits::eClosestHitKHR) {
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

    //Descriptor binding flags for bindless
    std::vector<vk::DescriptorBindingFlags> bindingFlags(bindings.size(), vk::DescriptorBindingFlags{});
    bindingFlags[5] = vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.setBindingFlags(bindingFlags);

    //Create descriptor set layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{};
    descSetLayoutInfo.setBindings(bindings);
    descSetLayoutInfo.setPNext(&bindingFlagsInfo);

    vk::UniqueDescriptorSetLayout descSetLayout = context->device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    //Descriptor set allocation with variable descriptor count
    uint32_t actualTextureCount = scene->textures.size();

    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountAllocInfo{};
    variableCountAllocInfo.setDescriptorCounts(actualTextureCount);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(context->descPool.get());
    allocInfo.setSetLayouts(*descSetLayout);
    allocInfo.setDescriptorSetCount(1);
    allocInfo.setPNext(&variableCountAllocInfo);

    std::vector<vk::UniqueDescriptorSet> descSets = context->device->allocateDescriptorSetsUnique(allocInfo);
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
    writes[0].setPNext(&scene->tlas->descAccelInfo);
    writes[1].setImageInfo(inputImage.descImageInfo);
    writes[2].setBufferInfo(scene->meshBuffer.getDescriptorInfo());
    writes[3].setBufferInfo(scene->materialBuffer.getDescriptorInfo());
    writes[4].setBufferInfo(scene->pointLightBuffer.getDescriptorInfo());
    writes[5].setImageInfo(scene->textureImageInfos);

    writes[5].setDescriptorCount(actualTextureCount);

    //Final descriptor update
    context->device->updateDescriptorSets(writes, {});

    //Create pipeline layout
    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(PushConstants));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    vk::UniquePipelineLayout pipelineLayout = context->device->createPipelineLayoutUnique(pipelineLayoutInfo);

    //Create pipeline
    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo;
    rtPipelineInfo.setStages(shaderStages);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(1);
    rtPipelineInfo.setLayout(*pipelineLayout);

    auto pipelineResult = context->device->createRayTracingPipelineKHRUnique({}, {}, rtPipelineInfo);
    if (pipelineResult.result != vk::Result::eSuccess)
        throw std::runtime_error("failed to create ray tracing pipeline.");

    vk::UniquePipeline pipeline = std::move(pipelineResult.value);

    //Get ray tracing properties
    auto properties = context->physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    //Calculate shader binding table (SBT) size
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = shaderGroups.size();
    uint32_t sbtSize = groupCount * handleSizeAligned;

    //Get shader group handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context->device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess)
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
    vk::StridedDeviceAddressRegionKHR raygenRegion{ raygenSBT.getDeviceAddress(), handleSizeAligned, raygenSize};
    vk::StridedDeviceAddressRegionKHR missRegion{ missSBT.getDeviceAddress(), handleSizeAligned, missSize };
    vk::StridedDeviceAddressRegionKHR hitRegion{ hitSBT.getDeviceAddress(), handleSizeAligned, hitSize };

    //CAMERA
    auto cam = std::make_unique<PerspectiveCamera>(
        scene,
        "Main Camera",
       Transform{},              // transform
        16.0f / 9.0f,          // aspect ratio
        36.0f,                 // sensor width in mm (typical full-frame)
        24.0f,                 // sensor height in mm
        50.0f,                 // focal length in mm
        2.8f,                  // aperture (f-stop)
        10.0f                  // focus distance in meters
    );

    scene->addSceneObject(std::move(cam));

    //Setup Input
    static bool isRayTracing = false;
    static int frame = 0;

    //Main loop
    uint32_t imageIndex = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;

    glfwSetMouseButtonCallback(context->window, [](GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_PRESS)
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); //capture cursor
            else if (action == GLFW_RELEASE)
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    });

    InputTracker inputTracker(context->window);
    HdrToLdrCompute hdrToLdrCompute(context->device.get(), inputImage.view.get(), outputImage.view.get());

    ImGuiManager imGuiManager(context, swapchainImages);

    // Create Ui components
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(scene->sceneObjects);
    auto viewportPanel = std::make_unique<ViewportPanel>(context, imGuiManager.getDescriptorPool(), outputImage.view.get());

    debugPanel->setSceneStats(0,scene->instances.size(),scene->pointLights.size());

    // Set up callbacks
    debugPanel->setModeChangedCallback([&]
    {
        isRayTracing = !isRayTracing;
        scene->markDirty();
    });

    mainMenuBar->setCallback("File.Quit", [&]() {
        glfwSetWindowShouldClose(context->window, GLFW_TRUE);
    });

    // Add components to manager
    imGuiManager.addComponent(std::move(mainMenuBar));
    imGuiManager.addComponent(std::move(debugPanel));
    imGuiManager.addComponent(std::move(outlinerDetailsPanel));
    imGuiManager.addComponent(std::move(viewportPanel));

    vk::UniqueSemaphore imageAcquiredSemaphore = context->device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    while (!glfwWindowShouldClose(context->window)) {

        //Delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        timeAccumulator += deltaTime;
        frameCounter++;

        if (timeAccumulator >= 1.0f) {
            //update FPS in UI
            if (auto* debugPanelPtr = dynamic_cast<DebugPanel*>(imGuiManager.getComponent("Debug")))
                debugPanelPtr->setFps(static_cast<float>(frameCounter) / timeAccumulator);
            //reset
            timeAccumulator = 0.0f;
            frameCounter = 0;
        }

        //Window Update
        glfwPollEvents();

        // ImGui Update
        imGuiManager.renderUi();

        //Camera Update
        inputTracker.update();
        scene->getActiveCamera()->update(inputTracker, deltaTime);

        //Push Constants
        PushConstants pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.push.isRayTracing = isRayTracing;
        pushConstantData.camera = scene->getActiveCamera()->getCameraData();

        //Acquire next image
        imageIndex = context->device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore).value;

        //Record commands
        vk::CommandBuffer commandBuffer = *commandBuffers[imageIndex];
        commandBuffer.begin(vk::CommandBufferBeginInfo());
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, {});
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR, 0, sizeof(PushConstants), &pushConstantData);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        //Convert 32bit float to 8bit image for display, read from inputImage and write to outputImage
        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
        hdrToLdrCompute.dispatch(commandBuffer, (WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);

        //Transition swapchain image layout to draw
        Image::setImageLayout(commandBuffer, swapchainImages[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        //Draw ImGui
        imGuiManager.Draw(commandBuffer, imageIndex);
        //Transition swapchain image layout to present
        Image::setImageLayout(commandBuffer, swapchainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

        //Submit
        commandBuffer.end();
        context->queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        //Present image
        vk::PresentInfoKHR presentInfo{};
        presentInfo.setSwapchains(*swapchain);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);

        auto result = context->queue.presentKHR(presentInfo);
        if (result != vk::Result::eSuccess)
            throw std::runtime_error("failed to present.");

        context->queue.waitIdle();

        //reset frame counter if something changed
        if (scene->getDirty())
            frame = 0;
        else
            frame++;

        //set to false for next frame
        scene->resetDirty();

    } //End Render Loop

    context->device->waitIdle();
    glfwDestroyWindow(context->window);
    glfwTerminate();
}