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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Utils.h"
#include "UI/EnvironmentPanel.h"

Viewer::Viewer(const int width, const int height)
    : width(width),
      height(height),
      context(width, height),
      scene(context),
      renderer(context, width, height),
      gpuRaytracer(context, scene, width /2, height /2),
      cpuRaytracer(context, scene, width /4, height /4),
      inputTracker(context.getWindow()),
      gpuImageTonemapper(context, width /2, height /2, gpuRaytracer.getOutputImage()),
      cpuImageTonemapper(context, width /4, height /4, cpuRaytracer.getOutputImage()),
      imGuiManager(context, renderer.getSwapchainImages(), width, height)
{
    setupScene();
    setupUI();
}

void Viewer::setupUI() {
    auto mainMenuBar = std::make_unique<MainMenuBar>();
    auto debugPanel = std::make_unique<DebugPanel>();
    auto environmentPanel = std::make_unique<EnvironmentPanel>(scene);
    auto outlinerDetailsPanel = std::make_unique<OutlinerDetailsPanel>(scene, inputTracker);
    auto gpuViewport = std::make_unique<ViewportPanel>(context, gpuImageTonemapper.getOutputImage(), width, height, "GPU Viewport");
    auto cpuViewport = std::make_unique<ViewportPanel>(context, cpuImageTonemapper.getOutputImage(), width, height, " CPU Viewport");

    mainMenuBar->setCallback("File.Quit", [&] {
        glfwSetWindowShouldClose(context.getWindow(), GLFW_TRUE);
    });

    // Import callback
    mainMenuBar->setCallback("File.Import.Obj", [this] {
        const auto selection = pfd::open_file("Import OBJ Model", ".", { "OBJ Files", "*.obj", "All Files", "*" }).result();
        if (!selection.empty()) {
            
            try {
                std::vector<Vertex> vertices;
                std::vector<uint32_t> indices;
                std::vector<Face> faces;
                std::vector<Material> materials;
                Utils::loadObj(scene, selection[0], vertices, indices, faces, materials);
                auto meshAsset = std::make_shared<MeshAsset>(scene, selection[0], std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
                
                this->scene.add(meshAsset);
                auto instance = std::make_unique<MeshInstance>(this->scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
                int instanceIndex = this->scene.add(std::move(instance));
                if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
                    outliner->setSelectedIndex(instanceIndex);

            } catch (const std::exception& e) {
                std::cerr << "Import failed: " << e.what() << std::endl;
            }
        }
    });

    mainMenuBar->setCallback("File.Import.CrtScene", [this] {
    const auto selection = pfd::open_file("Import CrtScene", ".", { "CRT Scene Files", "*.crtscene", "All Files", "*" }).result();
    if (!selection.empty()) {
            
        try {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            std::vector<Face> faces;
            std::vector<Material> materials;
            Utils::loadCrtScene(scene, selection[0], vertices, indices, faces, materials);
            auto meshAsset = std::make_shared<MeshAsset>(scene, selection[0], std::move(vertices), std::move(indices), std::move(faces), std::move(materials));
            this->scene.add(meshAsset);
            auto instance = std::make_unique<MeshInstance>(this->scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
            int instanceIndex = this->scene.add(std::move(instance));
            if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
                outliner->setSelectedIndex(instanceIndex);

        } catch (const std::exception& e) {
            std::cerr << "Import failed: " << e.what() << std::endl;
        }
    }
});

    // Add primitives callbacks
    mainMenuBar->setCallback("Add.Cube", [this] {
        auto cube = MeshAsset::CreateCube(scene, "Cube", {});
        this->scene.add(cube);
        auto instance = std::make_unique<MeshInstance>(this->scene, "Cube Instance", cube, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Plane", [this] {
        auto plane = MeshAsset::CreatePlane(this->scene, "Plane", {});
        this->scene.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->scene, "Plane Instance", plane, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.Sphere", [this] {
        auto sphere = MeshAsset::CreateSphere(this->scene, "Sphere", {}, 24, 48);
        this->scene.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->scene, "Sphere Instance", sphere, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.RectLight", [this] {
        Material material{};
        material.emission = glm::vec3(10);
        auto plane = MeshAsset::CreatePlane(this->scene, "RectLight", material);
        this->scene.add(plane);
        auto instance = std::make_unique<MeshInstance>(this->scene, "RectLight Instance", plane, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.SphereLight", [this] {
        Material material{};
        material.emission = glm::vec3(10);
        auto sphere = MeshAsset::CreateSphere(this->scene, "SphereLight", material, 24, 48);
        this->scene.add(sphere);
        auto instance = std::make_unique<MeshInstance>(this->scene, "SphereLight Instance", sphere, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    mainMenuBar->setCallback("Add.DiskLight", [this] {
        Material material{};
        material.emission = glm::vec3(10);
        auto disk = MeshAsset::CreateDisk(this->scene, "DiskLight", material, 48);
        this->scene.add(disk);
        auto instance = std::make_unique<MeshInstance>(this->scene, "DiskLight Instance", disk, Transform(glm::vec3(0, 0, 0)));
        int instanceIndex = this->scene.add(std::move(instance));
        if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
            outliner->setSelectedIndex(instanceIndex);
    });

    imGuiManager.add(std::move(mainMenuBar));
    imGuiManager.add(std::move(debugPanel));
    imGuiManager.add(std::move(environmentPanel));
    imGuiManager.add(std::move(outlinerDetailsPanel));
    imGuiManager.add(std::move(gpuViewport));
    imGuiManager.add(std::move(cpuViewport));
}

void Viewer::setupScene() {

    // Gray 1x1 texture
    scene.add(Texture(context, "Gray", (const uint8_t[]){127, 127, 127, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));

    // White 1x1 texture
    scene.add(Texture(context, "White", (const uint8_t[]){255, 255, 255, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));

    // Black 1x1 texture
    scene.add(Texture(context, "Black", (const uint8_t[]){0, 0, 0, 255}, 1, 1, vk::Format::eR8G8B8A8Unorm));
    
    // Add HDRI texture
    static constexpr unsigned char data[] = {
        #embed "../assets/Ultimate_Skies_4k_0036.hdr"
    };
    int imgWidth, imgHeight, channels;
    float* pixels = stbi_loadf_from_memory(data, static_cast<int>(sizeof(data)), &imgWidth, &imgHeight, &channels, 4);
    if (!pixels)
        throw std::runtime_error("Failed to load HDR texture from memory");

    scene.add(Texture(context, "HDRI Sky", pixels, imgWidth, imgHeight, vk::Format::eR32G32B32A32Sfloat));

    stbi_image_free(pixels);

    auto cam = std::make_unique<PerspectiveCamera>(scene, "Main Camera", Transform{glm::vec3(0, -1.0f, 3.5f), glm::vec3(-180, 0, 180), glm::vec3(0)}, width / static_cast<float>(height), 36.0f, 24.0f, 30.0f, 2.4f, 3.0f, 3.0f);
    scene.add(std::move(cam));

    auto meshAsset = MeshAsset::CreateCube(this->scene, "Default Cube", Material{});
    this->scene.add(meshAsset);
    auto instance = std::make_unique<MeshInstance>(this->scene, Utils::nameFromPath(meshAsset->getPath()) + " Instance", meshAsset, Transform{});
    int instanceIndex = this->scene.add(std::move(instance));
    if (auto* outliner = dynamic_cast<OutlinerDetailsPanel*>(this->imGuiManager.getComponent("Outliner Details")))
        outliner->setSelectedIndex(instanceIndex);
}

void Viewer::run() {
    using clock = std::chrono::high_resolution_clock;
    auto lastTime = clock::now();
    float timeAccumulator = 0.0f;
    int frameCounter = 0;
    
    while (!glfwWindowShouldClose(context.getWindow())) {
        vk::Fence inFlightFence = renderer.getCurrentInFlightFence();
        (void)context.getDevice().waitForFences(inFlightFence, VK_TRUE, UINT64_MAX);
        
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

        scene.getActiveCamera()->update(inputTracker, deltaTime);

        vk::Semaphore imageAcquiredSemaphore = renderer.getCurrentImageAcquiredSemaphore();
        auto resultValue = context.getDevice().acquireNextImageKHR(renderer.getSwapChain(), UINT64_MAX, imageAcquiredSemaphore);
        uint32_t imageIndex = resultValue.value;
        
        context.getDevice().resetFences(inFlightFence);
        
        if (scene.isTlasDirty() || scene.isMeshesDirty() || scene.isTexturesDirty()) {
            context.getDevice().waitIdle();
            if (scene.isMeshesDirty())
                gpuRaytracer.updateMeshes();
            if (scene.isTexturesDirty())
                gpuRaytracer.updateTextures();
            if (scene.isTlasDirty()) {
                gpuRaytracer.updateTLAS();
                cpuRaytracer.updateFromScene();
            }
        }
        
        if (scene.isAccumulationDirty())
            frame = 0;
        else
            frame++;
        
        scene.clearDirtyFlags();
        
        vk::CommandBuffer commandBuffer = renderer.getCommandBuffer(imageIndex);
        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        PushConstants pushConstantData{};
        pushConstantData.push.frame = frame;
        pushConstantData.camera = scene.getActiveCamera()->getCameraData();
        if (auto* environment = dynamic_cast<EnvironmentPanel*>(this->imGuiManager.getComponent("Environment")))
            pushConstantData.push.hdriTexture = environment->getHdriTexture();

        gpuRaytracer.render(commandBuffer, pushConstantData);
        cpuRaytracer.render(commandBuffer, pushConstantData);
        
        gpuImageTonemapper.dispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);
        cpuImageTonemapper.dispatch(commandBuffer, (width + 15) / 16, (height + 15) / 16, 1);
        
        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        imGuiManager.Draw(commandBuffer, imageIndex, width, height);
        Image::setImageLayout(commandBuffer, renderer.getSwapchainImages()[imageIndex], vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
        commandBuffer.end();

        vk::Semaphore renderFinishedSemaphore = renderer.getCurrentRenderFinishedSemaphore();
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submitInfo = vk::SubmitInfo()
            .setCommandBuffers(commandBuffer)
            .setWaitSemaphores(imageAcquiredSemaphore)
            .setWaitDstStageMask(waitStage)
            .setSignalSemaphores(renderFinishedSemaphore);
        context.getQueue().submit(submitInfo, inFlightFence);

        vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
            .setWaitSemaphores(renderFinishedSemaphore)
            .setSwapchains(renderer.getSwapChain())
            .setImageIndices(imageIndex);
        context.getQueue().presentKHR(presentInfo);

        renderer.advanceFrame();
    }

    context.getDevice().waitIdle();
}


Viewer::~Viewer()
{
    std::cout << "Destroying Viewer...." << std::endl;
}

void Viewer::cleanup()
{
    glfwTerminate();
}