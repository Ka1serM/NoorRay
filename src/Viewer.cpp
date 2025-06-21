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

Viewer::Viewer(int width, int height)
    : width(width),
      height(height),
      context(width, height),
      renderer(context, width, height),
      inputTracker(context.window),
      hdrToLdrCompute(context, width, height, renderer.inputImage.view.get()),
      imGuiManager(context, renderer.getSwapchainImages(), width, height)
{
    setupScene();
    setupUI();
}

Viewer::~Viewer() {
    stop();
}

void Viewer::stop() {
    if (!running.load())
        return;
    running = false;
}

void Viewer::runOnMainThread(std::function<void()> task)
{
    std::lock_guard<std::mutex> lock(mainThreadQueueMutex);
    mainThreadQueue.push(std::move(task));
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

        hdrToLdrCompute.dispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);

        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        imGuiManager.Draw(commandBuffer, imageIndex, width, height);
        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

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
    // Default PointLight for empty buffer
    PointLight pointLight{};
    pointLight.intensity = 0;
    renderer.add(pointLight);
    
    // Add HDRI texture
    renderer.add(Texture(context, R"(C:\Users\Marcel\Documents\GitRepositories\UniversalUmap\UniversalUmap.Rendering\assets\Ultimate_Skies_4k_0036.HDR)", vk::Format::eR32G32B32A32Sfloat));
    
    auto cam = std::make_unique<PerspectiveCamera>(renderer, "Main Camera", Transform{glm::vec3(0, -1.0f, 3.5f), glm::vec3(-180, 0, 180), glm::vec3(0)}, width / static_cast<float>(height), 36.0f, 24.0f, 30.0f, 16.0f, 3.0f);
    renderer.add(std::move(cam));
}

void Viewer::setupUI() {
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(renderer, inputTracker);
    auto viewportPanel = std::make_unique<ViewportPanel>(context, imGuiManager.getDescriptorPool(), hdrToLdrCompute.outputImage.view.get(), width, height);

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
                        auto meshAsset = MeshAsset::CreateFromObj(this->renderer, filePath);
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
        auto cube = MeshAsset::CreateCube(this->renderer, "Cube", {});
        this->renderer.add(cube);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "Cube Instance", cube, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Plane", [this] {
        auto plane = MeshAsset::CreatePlane(this->renderer, "Plane", {});
        this->renderer.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "Plane Instance", plane, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Sphere", [this] {
        auto sphere = MeshAsset::CreateSphere(this->renderer, "Sphere", {}, 16, 32);
        this->renderer.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "Sphere Instance", sphere, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.RectLight", [this] {
        Material material{};
        material.emission = glm::vec3(10);
        auto plane = MeshAsset::CreatePlane(this->renderer, "RectLight", material);
        this->renderer.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "RectLight Instance", plane, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.SphereLight", [this] {
        Material material{};
        material.emission = glm::vec3(10);
        auto sphere = MeshAsset::CreateSphere(this->renderer, "SphereLight", material, 16, 32);
        this->renderer.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "SphereLight Instance", sphere, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.DiskLight", [this] {
    Material material{};
    material.emission = glm::vec3(10);
    auto disk = MeshAsset::CreateDisk(this->renderer, "DiskLight", material, 32);
    this->renderer.add(disk);
    auto instance = std::make_unique<MeshInstance>(this->renderer, "DiskLight Instance", disk, Transform(glm::vec3(0, 0, 0)));
    int instanceIndex = this->renderer.add(std::move(instance));
    if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
        outliner->setSelectedIndex(instanceIndex);
    });
    
    imGuiManager.addComponent(std::move(mainMenuBar));
    imGuiManager.addComponent(std::move(debugPanel));
    imGuiManager.addComponent(std::move(outlinerDetailsPanel));
    imGuiManager.addComponent(std::move(viewportPanel));
}