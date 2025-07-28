#pragma once

#include <vector>
#include "Scene/MeshInstance.h"
#include "Shaders/SharedStructs.h"
#include "Vulkan/Context.h"

class CpuRaytracer {
public:
    CpuRaytracer(Context& context, Scene& scene, uint32_t width, uint32_t height);
    void render(const vk::CommandBuffer& commandBuffer, const PushConstants& pushConstants);
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
    std::vector<glm::vec4> colorBuffer;
    
    std::vector<const MeshInstance*> threadSafeInstances;
};