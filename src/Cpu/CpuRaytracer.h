#pragma once

#include <future>
#include <vector>
#include "Scene/MeshInstance.h"
#include "../Shaders/PathTracing/SharedStructs.h"
#include "Vulkan/Context.h"

class CpuRaytracer {
public:
    CpuRaytracer(Context& context, Scene& scene, uint32_t width, uint32_t height);
    void render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants);

    // Async render API
    void startAsyncRender(const PushConstants& pushConstants);
    bool isRenderComplete() const;
    void waitForRender();
    void retrieveRenderResult(); // Uploads colorBuffer to outputImage

    // Output Vulkan image
    Image& getOutputImage() { return outputImage; }
    
    void updateFromScene();

private:
    void raygen(int x, int y, const PushConstants& pushConstants, Payload& payload);
    void traceRayEXT_CPU(Payload& payload, float tMin, float tMax);
    void closesthit(const HitInfo& hit, Payload& payload);
    void miss(Payload& payload);

    Context& context;
    Scene& scene;
    uint32_t width, height;
    Image outputImage;
    std::vector<vec4> colorBuffer;
    
    std::vector<const MeshInstance*> threadSafeInstances;

    // Async support
    std::future<void> renderFuture;
    std::atomic<bool> renderInProgress = false;
    PushConstants lastPushConstants;
};