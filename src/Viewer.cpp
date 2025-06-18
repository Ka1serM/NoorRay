#include "Viewer.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

#include "UI/DebugPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/OutlinerDetailsPanel.h"
#include "UI/ViewportPanel.h"
#include <GLFW/glfw3native.h>
#include "portable-file-dialogs.h"
#include "Utils.h"

HWND Viewer::getHwnd() const {
    return glfwGetWin32Window(context.window);
}

Viewer::Viewer(int width, int height)
    : width(width),
      height(height),
      context(width, height),
      renderer(context, width, height),
      inputImage(context, width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputImage(context, width, height, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      inputTracker(context.window),
      hdrToLdrCompute(context.device.get(), inputImage.view.get(), outputImage.view.get()),
      imGuiManager(context, renderer.getSwapchainImages(), width, height)
{
    renderer.updateStorageImage(inputImage.view.get());
    setupScene();
    setupUI();
}

Viewer::~Viewer() {
    stop();
}

void Viewer::stop() {
    if (!running.load()) return;
    running = false;
}

void Viewer::runOnMainThread(std::function<void()> task)
{
    std::lock_guard<std::mutex> lock(mainThreadQueueMutex);
    mainThreadQueue.push(std::move(task));
}

void Viewer::addMesh(const MeshData& mesh) {
    
}

void Viewer::start() {
    if (running.load())
        return;
    running = true;
    
    using clock = std::chrono::high_resolution_clock;
    auto lastTime = clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;

    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());

    while (running && !glfwWindowShouldClose(context.window)) {
        auto currentTime = clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        timeAccumulator += deltaTime;
        frameCounter++;

        if (timeAccumulator >= 1.0f) {
            if (auto* debugPanelPtr = dynamic_cast<DebugPanel*>(imGuiManager.getComponent("Debug")))
                debugPanelPtr->setFps(static_cast<float>(frameCounter) / timeAccumulator);

            timeAccumulator = 0.0f;
            frameCounter = 0;
        }

        {
            std::lock_guard<std::mutex> lock(mainThreadQueueMutex);
            while (!mainThreadQueue.empty()) {
                auto task = std::move(mainThreadQueue.front());
                mainThreadQueue.pop();
                task();
            }
        }

        glfwPollEvents();
        imGuiManager.renderUi();
        inputTracker.update();

        renderer.getActiveCamera()->update(inputTracker, deltaTime);
        
        PushConstants pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.push.isRayTracing = isRayTracing;
        pushConstantData.camera = renderer.getActiveCamera()->getCameraData();

        uint32_t imageIndex = context.device->acquireNextImageKHR(renderer.getSwapChain(), UINT64_MAX, *imageAcquiredSemaphore).value;

        renderer.render(imageIndex, pushConstantData);

        vk::CommandBuffer commandBuffer = renderer.getCommandBuffer(imageIndex);

        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eGeneral);
        hdrToLdrCompute.dispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);
        outputImage.setImageLayout(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal);

        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eUndefined,
                             vk::ImageLayout::eColorAttachmentOptimal);
        imGuiManager.Draw(commandBuffer, imageIndex, width, height);
        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                             vk::ImageLayout::ePresentSrcKHR);

        commandBuffer.end();

        context.queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        vk::PresentInfoKHR presentInfo{};
        presentInfo.setPSwapchains(&renderer.getSwapChain());
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);

        auto result = context.queue.presentKHR(presentInfo);
        if (result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to present.");

        context.queue.waitIdle();

        if (renderer.getDirty())
            frame = 0;
        else
            frame++;

        renderer.resetDirty();
    }

    context.device->waitIdle();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}

void Viewer::setupScene() {
    PointLight pointLight{};
    renderer.add(pointLight);

    // Material with emission
    Material material{};
    material.albedo = glm::vec3(1.0f);
    material.emission = glm::vec3(10.0f);
    renderer.add(material);

    // White 1x1 texture (one pixel white)
    renderer.add(Texture(context, (const uint8_t[]){255, 255, 255, 255}, 1, 1));
    
    auto cam = std::make_unique<PerspectiveCamera>(renderer, "Main Camera", Transform{glm::vec3(0), glm::vec3(0), glm::vec3(0)}, width / static_cast<float>(height), 36.0f, 24.0f, 50.0f, 2.8f, 10.0f);
    renderer.add(std::move(cam));
}

void Viewer::setupUI() {
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(renderer, inputTracker);
    auto viewportPanel = std::make_unique<ViewportPanel>(context, imGuiManager.getDescriptorPool(), outputImage.view.get(), width, height);

    debugPanel->setSceneStats(0, 0, 0);

    debugPanel->setModeChangedCallback([&] {
        isRayTracing = !isRayTracing;
        renderer.markDirty();
    });

    mainMenuBar->setCallback("File.Quit", [&] {
        glfwSetWindowShouldClose(context.window, GLFW_TRUE);
    });

    //Import callback
    mainMenuBar->setCallback("File.Import", [this] {
        std::thread([this] {
            auto selection = pfd::open_file("Import OBJ Model", ".", { "OBJ Files", "*.obj", "All Files", "*" }).result();
            if (!selection.empty()) {
                std::string filePath = selection[0];
                this->runOnMainThread([this, filePath]() {
                    try {
                        auto meshAsset = MeshAsset::CreateFromObj(this->context, this->renderer, filePath);
                        this->renderer.add(meshAsset);
                        auto instance = std::make_unique<MeshInstance>(this->renderer, Utils::nameFromPath(meshAsset->path) + " Instance", meshAsset, Transform(glm::vec3(0, 0, 0)));
                        int instanceIndex = this->renderer.add(std::move(instance));
                        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
                            outliner->setSelectedIndex(instanceIndex);
                    } catch (const std::exception& e) {
                        std::cerr << "Import failed: " << e.what() << std::endl;
                    }
                });
            }
        }).detach();
    });

    //Add primitives callbacks
    mainMenuBar->setCallback("Add.Cube", [this] {
        auto cube = MeshAsset::CreateCube(this->context, "Cube");
        this->renderer.add(cube);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "Cube Instance", cube, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Plane", [this] {
        auto plane = MeshAsset::CreatePlane(this->context, "Plane");
        this->renderer.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "Plane Instance", plane, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Sphere", [this] {
        auto sphere = MeshAsset::CreateSphere(this->context, "Sphere", 16, 32);
        this->renderer.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "Sphere Instance", sphere, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });
    
    imGuiManager.addComponent(std::move(mainMenuBar));
    imGuiManager.addComponent(std::move(debugPanel));
    imGuiManager.addComponent(std::move(outlinerDetailsPanel));
    imGuiManager.addComponent(std::move(viewportPanel));
}