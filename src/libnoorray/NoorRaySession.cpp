#include "NoorRaySession.h"

#include "Backend/OptiX/Runtime/Raytracer.h"
#include "Backend/Vulkan/Runtime/Renderer.h"

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
