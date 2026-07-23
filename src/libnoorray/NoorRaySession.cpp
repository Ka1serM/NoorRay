#include "NoorRaySession.h"

#include "Raytracing/Runtime/Raytracer.h"
#include "Vulkan/Renderer.h"

namespace noorray
{

NoorRaySession::NoorRaySession()
    : scene(context)
{
    raytracer = std::make_unique<Raytracer>(context, scene);
}

NoorRaySession::NoorRaySession(VulkanSurfaceProvider& surfaceProvider)
    : context(surfaceProvider)
    , scene(context)
    , headless(false)
{
    raytracer = std::make_unique<Raytracer>(context, scene);
    renderer = std::make_unique<Renderer>(
        context, surfaceProvider.getWidth(), surfaceProvider.getHeight());
}

NoorRaySession::~NoorRaySession() = default;

}
