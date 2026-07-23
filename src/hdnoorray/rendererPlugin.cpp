#include "rendererPlugin.h"

#include "renderDelegate.h"

#include <pxr/imaging/hd/rendererPluginRegistry.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType)
{
    HdRendererPluginRegistry::Define<HdNoorRayRendererPlugin>();
}

HdRenderDelegate* HdNoorRayRendererPlugin::CreateRenderDelegate()
{
    return new HdNoorRayRenderDelegate();
}

HdRenderDelegate* HdNoorRayRendererPlugin::CreateRenderDelegate(
    const HdRenderSettingsMap& settingsMap)
{
    return new HdNoorRayRenderDelegate(settingsMap);
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
