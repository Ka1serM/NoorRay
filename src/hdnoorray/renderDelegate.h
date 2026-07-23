#pragma once

#include "api.h"

#include <pxr/imaging/hd/renderDelegate.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdNoorRayRenderParam;

class HDNOORRAY_API HdNoorRayRenderDelegate final : public HdRenderDelegate
{
public:
    HdNoorRayRenderDelegate();
    explicit HdNoorRayRenderDelegate(const HdRenderSettingsMap& settingsMap);
    ~HdNoorRayRenderDelegate() override;

    const TfTokenVector& GetSupportedRprimTypes() const override;
    const TfTokenVector& GetSupportedSprimTypes() const override;
    const TfTokenVector& GetSupportedBprimTypes() const override;
    HdRenderParam* GetRenderParam() const override;
    HdResourceRegistrySharedPtr GetResourceRegistry() const override;

    HdRenderPassSharedPtr CreateRenderPass(
        HdRenderIndex* index, const HdRprimCollection& collection) override;

    HdInstancer* CreateInstancer(
        HdSceneDelegate* delegate, const SdfPath& id) override;
    void DestroyInstancer(HdInstancer* instancer) override;

    HdRprim* CreateRprim(const TfToken& typeId, const SdfPath& rprimId) override;
    void DestroyRprim(HdRprim* rprim) override;

    HdSprim* CreateSprim(const TfToken& typeId, const SdfPath& sprimId) override;
    HdSprim* CreateFallbackSprim(const TfToken& typeId) override;
    void DestroySprim(HdSprim* sprim) override;

    HdBprim* CreateBprim(const TfToken& typeId, const SdfPath& bprimId) override;
    HdBprim* CreateFallbackBprim(const TfToken& typeId) override;
    void DestroyBprim(HdBprim* bprim) override;

    void CommitResources(HdChangeTracker* tracker) override;
    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;
    HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;
    VtDictionary GetRenderStats() const override;

private:
    std::unique_ptr<HdNoorRayRenderParam> renderParam_;
    HdResourceRegistrySharedPtr resourceRegistry_;
};

PXR_NAMESPACE_CLOSE_SCOPE
