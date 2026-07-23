#pragma once

#include "api.h"

#include <pxr/imaging/hd/rendererPlugin.h>

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayRendererPlugin final : public HdRendererPlugin
{
public:
    HdRenderDelegate* CreateRenderDelegate() override;
    HdRenderDelegate* CreateRenderDelegate(
        const HdRenderSettingsMap& settingsMap) override;
    void DeleteRenderDelegate(HdRenderDelegate* renderDelegate) override;

    bool IsSupported(
        const HdRendererCreateArgs& rendererCreateArgs,
        std::string* reasonWhyNot = nullptr) const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
