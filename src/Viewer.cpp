#include "Viewer.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "UI/DebugPanel.h"
#include "UI/MainMenuBar.h"
#include "UI/OutlinerDetailsPanel.h"
#include "UI/ViewportPanel.h"
#include "portable-file-dialogs.h"
#include "stb_image.h"
#include "Utils.h"
#include "UI/EnvironmentPanel.h"

Viewer::Viewer(int width, int height)
    : width(width),
      height(height),
      context(width, height),
      renderer(context, width, height),
      inputTracker(context.window),
      tonemapper(context, width, height, renderer.inputImage.view.get()),
      imGuiManager(context, renderer.getSwapchainImages(), width, height)
{
    setupScene();
    setupUI();
}

void Viewer::setupUI() {
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto environmentPanel = std::make_unique<EnvironmentPanel>(renderer);
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(renderer, inputTracker);
    auto viewportPanel = std::make_unique<ViewportPanel>(context, tonemapper.outputImage.view.get(), width, height);

    mainMenuBar->setCallback("File.Quit", [&] {
        glfwSetWindowShouldClose(context.window, GLFW_TRUE);
    });

    // Import callback
    mainMenuBar->setCallback("File.Import", [this] {
        const auto selection = pfd::open_file("Import OBJ Model", ".", { "OBJ Files", "*.obj", "All Files", "*" }).result();
        if (!selection.empty()) {
            
            try {
                auto meshAsset = MeshAsset::CreateFromObj(this->renderer, selection[0]);
                this->renderer.add(meshAsset);
                auto instance = std::make_unique<MeshInstance>(this->renderer, Utils::nameFromPath(meshAsset->path) + " Instance", meshAsset, Transform{});
                int instanceIndex = this->renderer.add(std::move(instance));
                if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
                    outliner->setSelectedIndex(instanceIndex);

            } catch (const std::exception& e) {
                std::cerr << "Import failed: " << e.what() << std::endl;
            }
        }
    });

    // Add primitives callbacks
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
        auto sphere = MeshAsset::CreateSphere(this->renderer, "Sphere", {}, 24, 48);
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
        auto sphere = MeshAsset::CreateSphere(this->renderer, "SphereLight", material, 24, 48);
        this->renderer.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "SphereLight Instance", sphere, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.DiskLight", [this] {
        Material material{};
        material.emission = glm::vec3(10);
        auto disk = MeshAsset::CreateDisk(this->renderer, "DiskLight", material, 48);
        this->renderer.add(disk);
        auto instance = std::make_unique<MeshInstance>(this->renderer, "DiskLight Instance", disk, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->renderer.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    imGuiManager.add(std::move(mainMenuBar));
    imGuiManager.add(std::move(debugPanel));
    imGuiManager.add(std::move(environmentPanel));
    imGuiManager.add(std::move(outlinerDetailsPanel));
    imGuiManager.add(std::move(viewportPanel));
}

void Viewer::setupScene() {
    // Add HDRI texture
    static constexpr unsigned char data[] = {
        #embed "../assets/Ultimate_Skies_4k_0036.hdr"
    };
    int imgWidth, imgHeight, channels;
    float* pixels = stbi_loadf_from_memory(data, static_cast<int>(sizeof(data)), &imgWidth, &imgHeight, &channels, 4);
    if (!pixels)
        throw std::runtime_error("Failed to load HDR texture from memory");

    renderer.add(Texture(context, "HDRI Sky", pixels, imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));

    stbi_image_free(pixels);

    // White 1x1 texture
    renderer.add(Texture(context, "White", (const uint8_t[]){255, 255, 255, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));

    // Black 1x1 texture
    renderer.add(Texture(context, "Black", (const uint8_t[]){0, 0, 0, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));

    // Gray 1x1 texture
    renderer.add(Texture(context, "Gray", (const uint8_t[]){127, 127, 127, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));

    auto cam = std::make_unique<PerspectiveCamera>(renderer, "Main Camera", Transform{glm::vec3(0, -1.0f, 3.5f), glm::vec3(-180, 0, 180), glm::vec3(0)}, width / static_cast<float>(height), 36.0f, 24.0f, 30.0f, 2.4f, 3.0f, 3.0f);
    renderer.add(std::move(cam));
}

void Viewer::run() {
    
    using clock = std::chrono::high_resolution_clock;
    auto lastTime = clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;

    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());

    while (!glfwWindowShouldClose(context.window)) {
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

        glfwPollEvents();
        imGuiManager.renderUi();
        inputTracker.update();

        renderer.getActiveCamera()->update(inputTracker, deltaTime);

        PushConstants pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.camera = renderer.getActiveCamera()->getCameraData();

        if (auto* environment = dynamic_cast<EnvironmentPanel*>(this->imGuiManager.getComponent("Environment")))
            pushConstantData.push.hdriTexture = environment->getHdriTexture();

        uint32_t imageIndex = context.device->acquireNextImageKHR(renderer.getSwapChain(), UINT64_MAX, *imageAcquiredSemaphore).value;

        renderer.render(imageIndex, pushConstantData);

        vk::CommandBuffer commandBuffer = renderer.getCommandBuffer(imageIndex);

        tonemapper.dispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);

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
}

Viewer::~Viewer()
{
    std::cout << "Destroying Viewer...." << std::endl;
}

void Viewer::cleanup()
{
    glfwTerminate();
}