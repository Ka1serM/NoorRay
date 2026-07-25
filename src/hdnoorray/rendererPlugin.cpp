#include "rendererPlugin.h"

#include "renderDelegate.h"

#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>

#include <exception>

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType)
{
    HdRendererPluginRegistry::Define<HdNoorRayRendererPlugin>();
}

// Creating a delegate brings up Vulkan, CUDA and OptiX. If any of that fails
// the host gets a null delegate and falls back to another renderer, rather than
// an exception escaping across the plugin boundary.
HdRenderDelegate* HdNoorRayRendererPlugin::CreateRenderDelegate()
{
    try {
        return new HdNoorRayRenderDelegate();
    } catch (const std::exception& error) {
        TF_RUNTIME_ERROR(
            "hdNoorRay could not initialize its renderer: %s", error.what());
        return nullptr;
    }
}

HdRenderDelegate* HdNoorRayRendererPlugin::CreateRenderDelegate(
    const HdRenderSettingsMap& settingsMap)
{
    try {
        return new HdNoorRayRenderDelegate(settingsMap);
    } catch (const std::exception& error) {
        TF_RUNTIME_ERROR(
            "hdNoorRay could not initialize its renderer: %s", error.what());
        return nullptr;
    }
}

void HdNoorRayRendererPlugin::DeleteRenderDelegate(
    HdRenderDelegate* renderDelegate)
{
    delete renderDelegate;
}

bool HdNoorRayRendererPlugin::IsSupported(
    const HdRendererCreateArgs&, std::string*) const
{
    return true;
}

PXR_NAMESPACE_CLOSE_SCOPE
