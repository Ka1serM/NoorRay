#pragma once

#include <memory>

#include "Scene/Scene.h"
#include "Backend/Vulkan/Runtime/Context.h"

class Raytracer;
class Renderer;

namespace noorray
{

class NoorRaySession
{
public:
    NoorRaySession();
    explicit NoorRaySession(VulkanSurfaceProvider& surfaceProvider);
    ~NoorRaySession();

    NoorRaySession(const NoorRaySession&) = delete;
    NoorRaySession& operator=(const NoorRaySession&) = delete;

    Context context;
    Scene scene;
    std::unique_ptr<Raytracer> raytracer;
    std::unique_ptr<Renderer> renderer;
    bool headless{true};
};

}
