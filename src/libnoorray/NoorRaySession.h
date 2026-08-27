#pragma once

#include <memory>
#include <cstdint>

#include "Scene/Scene.h"
#include <gpu/gpu.hpp>
#include "Materials/MaterialX/MaterialXSceneRuntime.h"

class VulkanRaytracer;

namespace noorray
{

class NoorRaySession
{
public:
    NoorRaySession();
    explicit NoorRaySession(gpu::SurfaceProvider& surfaceProvider);
    ~NoorRaySession();

    // Publishes the current scene into the native Vulkan renderer. Hosts call
    // this after scene import or a batch of geometry edits; replacement is
    // immutable from the dispatcher's point of view.
    void rebuildNativeScene();
    // Publishes scene edits made by the editor after startup. Returns true
    // when a GPU snapshot changed and accumulation must restart.
    bool pollNativeScene();
    void rebuildNativeMaterials();
    // Advances asynchronous MaterialX compilation without blocking the UI and
    // publishes a new immutable GPU snapshot only after the whole table is ready.
    bool pollNativeMaterials();
    void updateNativeCamera();
    // Creates the headless gpu::Device/VulkanRaytracer pair used by embedded
    // hosts such as Python and Hydra. It is deliberately explicit so merely
    // constructing a scene remains GPU-free.
    void initializeHeadlessRenderer(uint32_t width, uint32_t height,
        bool exportColorMemory = false);

    NoorRaySession(const NoorRaySession&) = delete;
    NoorRaySession& operator=(const NoorRaySession&) = delete;

    Scene scene;
    std::unique_ptr<gpu::Device> device;
    std::unique_ptr<gpu::Swapchain> swapchain;
    // Native Vulkan raytracer path.
    std::unique_ptr<VulkanRaytracer> raytracer;
    bool headless{true};
    // Owns the immutable host-side MaterialX -> SVM compilation snapshot
    // consumed by the Vulkan raytracer dispatch.
    MaterialXSceneRuntime materialRuntime;

private:
    std::size_t publishedMaterialWordCount{};
};

}
