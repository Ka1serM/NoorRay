#include <chrono>
#include <vector>

#include "Vulkan/Context.h"
#include "Vulkan/Image.h"
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
#include "Vulkan/Renderer.h"

int main() {
    Context context(WIDTH, HEIGHT);

    Renderer renderer{context};

    Image inputImage{context, WIDTH, HEIGHT, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};
    renderer.updateStorageImage(inputImage.view.get());

    Image outputImage{context,WIDTH, HEIGHT, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst };

    Scene scene{};

    PointLight pointLight{};
    renderer.add(pointLight);

    auto meshAsset2 = std::make_shared<MeshAsset>(context, renderer, "../assets/LittleTokyo.obj");
    scene.addMeshAsset(meshAsset2);

    auto meshInstance = std::make_unique<MeshInstance>(renderer, "Mesh Instance 1", meshAsset2, Transform(glm::vec3(0, 100, 0), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1)));
    auto meshInstance2 = std::make_unique<MeshInstance>(renderer, "Instance 2", meshAsset2, Transform());
    scene.addMeshInstance(std::move(meshInstance));
    scene.addMeshInstance(std::move(meshInstance2));

    //CAMERA
    auto cam = std::make_unique<PerspectiveCamera>(renderer,"Main Camera", Transform{}, WIDTH / HEIGHT, 36.0f, 24.0f, 50.0f, 2.8f, 10.0f);
    scene.addSceneObject(std::move(cam));

    //Setup Input
    static bool isRayTracing = false;
    static int frame = 0;

    //Main loop
    auto lastTime = std::chrono::high_resolution_clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;

    InputTracker inputTracker(context.window);
    HdrToLdrCompute hdrToLdrCompute(context.device.get(), inputImage.view.get(), outputImage.view.get());

    ImGuiManager imGuiManager(context, renderer.getSwapchainImages());

    // Create Ui components
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(scene.sceneObjects);
    auto viewportPanel = std::make_unique<ViewportPanel>(context, imGuiManager.getDescriptorPool(), outputImage.view.get());

    debugPanel->setSceneStats(0,0,0);

    // Set up callbacks
    debugPanel->setModeChangedCallback([&]
    {
        isRayTracing = !isRayTracing;
        renderer.markDirty();
    });

    mainMenuBar->setCallback("File.Quit", [&]
    {
        glfwSetWindowShouldClose(context.window, GLFW_TRUE);
    });

    // Add components to manager
    imGuiManager.addComponent(std::move(mainMenuBar));
    imGuiManager.addComponent(std::move(debugPanel));
    imGuiManager.addComponent(std::move(outlinerDetailsPanel));
    imGuiManager.addComponent(std::move(viewportPanel));

    uint32_t imageIndex = 0;
    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    while (!glfwWindowShouldClose(context.window)) {

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
        scene.getActiveCamera()->update(inputTracker, deltaTime);

        //Push Constants
        PushConstants pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.push.isRayTracing = isRayTracing;
        pushConstantData.camera = scene.getActiveCamera()->getCameraData();

        //Acquire next image
        imageIndex = context.device->acquireNextImageKHR(renderer.getSwapChain(), UINT64_MAX, *imageAcquiredSemaphore).value;

        //Record commands
        renderer.render(imageIndex, pushConstantData);

        vk::CommandBuffer commandBuffer = renderer.getCommandBuffer(imageIndex);
        //Convert 32bit float to 8bit image for display, read from inputImage and write to outputImage
        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
        hdrToLdrCompute.dispatch(commandBuffer, (WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);

        //Transition swapchain image layout to draw
        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        //Draw ImGui
        imGuiManager.Draw(commandBuffer, imageIndex);
        //Transition swapchain image layout to present
        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

        //Submit
        commandBuffer.end();
        context.queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        //Present image
        vk::PresentInfoKHR presentInfo{};
        vk::SwapchainKHR rawSwapchain = renderer.getSwapChain();
        presentInfo.setPSwapchains(&rawSwapchain);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);

        auto result = context.queue.presentKHR(presentInfo);
        if (result != vk::Result::eSuccess)
            throw std::runtime_error("failed to present.");

        context.queue.waitIdle();

        //reset frame counter if something changed
        if (renderer.getDirty())
            frame = 0;
        else
            frame++;

        //set to false for next frame
        renderer.resetDirty();

    } //End Render Loop

    context.device->waitIdle();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}