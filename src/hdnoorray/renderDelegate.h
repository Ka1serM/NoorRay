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
    void SetRenderSetting(
        const TfToken& key, const VtValue& value) override;
    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;
    HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;
    VtDictionary GetRenderStats() const override;
    bool IsParallelSyncEnabled(const TfToken& primType) const override;

    // Without this, Hydra hands HdNoorRayMaterial::Sync the *auto-derived*
    // UsdPreviewSurface fallback USD's own material-adapter machinery
    // synthesizes for any consumer that doesn't ask for the real MaterialX
    // network (see docs/MaterialX.md's "Defects fixed" section) -- never
    // the actual node graph, no matter what Blender's own
    // bl_use_materialx export produced. "mtlx" is the standard token
    // Hydra/USD render delegates use to request that network (matching
    // Storm, RenderMan's HdPrman, Arnold's HdArnold, ...); the trailing
    // empty token keeps the universal/preview network as a fallback for
    // materials that have no MaterialX network at all.
    TfTokenVector GetMaterialRenderContexts() const override;

private:
    std::unique_ptr<HdNoorRayRenderParam> renderParam_;
    HdResourceRegistrySharedPtr resourceRegistry_;
};

PXR_NAMESPACE_CLOSE_SCOPE
