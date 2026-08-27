#include "renderDelegate.h"

#include "mesh.h"
#include "instancer.h"
#include "lights/diskLight.h"
#include "lights/distantLight.h"
#include "lights/domeLight.h"
#include "lights/rectLight.h"
#include "lights/sphereLight.h"
#include "material.h"
#include "renderBuffer.h"
#include "renderParam.h"
#include "renderPass.h"

#include <pxr/imaging/hd/bprim.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sprim.h>
#include <pxr/imaging/hd/tokens.h>

#include <pxr/base/gf/vec4f.h>

#include <string_view>
#include <vector>

#if PXR_VERSION < 2508 || PXR_VERSION > 2603
#error "hdNoorRay supports the OpenUSD 25.08 and 26.03 ABIs used by Blender 5.2"
#endif

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
const TfTokenVector RprimTypes{HdPrimTypeTokens->mesh};
const TfTokenVector SprimTypes{
    HdPrimTypeTokens->camera,
    HdPrimTypeTokens->material,
    HdPrimTypeTokens->sphereLight,
    HdPrimTypeTokens->diskLight,
    HdPrimTypeTokens->rectLight,
    HdPrimTypeTokens->distantLight,
    HdPrimTypeTokens->domeLight,
};
const TfTokenVector BprimTypes{HdPrimTypeTokens->renderBuffer};

constexpr std::string_view MaterialXSettingPrefix = "noorray:materialx:";

std::string_view MaterialXMaterialLeaf(const TfToken& key)
{
    const std::string& text = key.GetString();
    if (!std::string_view(text).starts_with(MaterialXSettingPrefix))
        return {};
    return std::string_view(text).substr(MaterialXSettingPrefix.size());
}

bool StoreMaterialXSetting(HdNoorRayRenderParam& param,
    const TfToken& key, const VtValue& value)
{
    const std::string_view materialLeaf = MaterialXMaterialLeaf(key);
    if (materialLeaf.empty() || !value.IsHolding<std::string>())
        return false;
    return param.SetMaterialXDocument(
        std::string(materialLeaf), value.UncheckedGet<std::string>());
}
}

HdNoorRayRenderDelegate::HdNoorRayRenderDelegate()
    : renderParam_(std::make_unique<HdNoorRayRenderParam>())
    , resourceRegistry_(std::make_shared<HdResourceRegistry>())
{
}

HdNoorRayRenderDelegate::HdNoorRayRenderDelegate(
    const HdRenderSettingsMap& settingsMap)
    : HdRenderDelegate(settingsMap)
    , renderParam_(std::make_unique<HdNoorRayRenderParam>())
    , resourceRegistry_(std::make_shared<HdResourceRegistry>())
{
    // Construction-time settings bypass the virtual SetRenderSetting path.
    // Extract transport documents once, then remove the large XML values from
    // HdRenderDelegate's generic settings map so thousands of materials do
    // not remain stored twice for the delegate lifetime.
    std::vector<TfToken> materialXKeys;
    for (const auto& [key, value] : settingsMap) {
        if (!MaterialXMaterialLeaf(key).empty()) {
            StoreMaterialXSetting(*renderParam_, key, value);
            materialXKeys.push_back(key);
        }
    }
    for (const TfToken& key : materialXKeys)
        _settingsMap.erase(key);
}

HdNoorRayRenderDelegate::~HdNoorRayRenderDelegate() = default;

const TfTokenVector& HdNoorRayRenderDelegate::GetSupportedRprimTypes() const
{
    return RprimTypes;
}

const TfTokenVector& HdNoorRayRenderDelegate::GetSupportedSprimTypes() const
{
    return SprimTypes;
}

const TfTokenVector& HdNoorRayRenderDelegate::GetSupportedBprimTypes() const
{
    return BprimTypes;
}

HdRenderParam* HdNoorRayRenderDelegate::GetRenderParam() const
{
    return renderParam_.get();
}

HdResourceRegistrySharedPtr HdNoorRayRenderDelegate::GetResourceRegistry() const
{
    return resourceRegistry_;
}

HdRenderPassSharedPtr HdNoorRayRenderDelegate::CreateRenderPass(
    HdRenderIndex* index, const HdRprimCollection& collection)
{
    return std::make_shared<HdNoorRayRenderPass>(
        index, collection, *renderParam_);
}

HdInstancer* HdNoorRayRenderDelegate::CreateInstancer(
    HdSceneDelegate* delegate, const SdfPath& id)
{
    return new HdNoorRayInstancer(delegate, id);
}

void HdNoorRayRenderDelegate::DestroyInstancer(HdInstancer* instancer)
{
    delete instancer;
}

HdRprim* HdNoorRayRenderDelegate::CreateRprim(
    const TfToken& typeId, const SdfPath& id)
{
    if (typeId == HdPrimTypeTokens->mesh) {
        return new HdNoorRayMesh(id);
    }
    return nullptr;
}

void HdNoorRayRenderDelegate::DestroyRprim(HdRprim* rprim)
{
    delete rprim;
}

HdSprim* HdNoorRayRenderDelegate::CreateSprim(
    const TfToken& typeId, const SdfPath& id)
{
    if (typeId == HdPrimTypeTokens->camera)
        return new HdCamera(id);
    if (typeId == HdPrimTypeTokens->material)
        return new HdNoorRayMaterial(id);
    if (typeId == HdPrimTypeTokens->sphereLight)
        return new HdNoorRaySphereLight(id);
    if (typeId == HdPrimTypeTokens->rectLight)
        return new HdNoorRayRectLight(id);
    if (typeId == HdPrimTypeTokens->diskLight)
        return new HdNoorRayDiskLight(id);
    if (typeId == HdPrimTypeTokens->distantLight)
        return new HdNoorRayDistantLight(id);
    if (typeId == HdPrimTypeTokens->domeLight)
        return new HdNoorRayDomeLight(id);
    return nullptr;
}

HdSprim* HdNoorRayRenderDelegate::CreateFallbackSprim(const TfToken& typeId)
{
    if (typeId == HdPrimTypeTokens->camera)
        return new HdCamera(SdfPath());
    if (typeId == HdPrimTypeTokens->material)
        return new HdNoorRayMaterial(SdfPath());
    if (typeId == HdPrimTypeTokens->sphereLight)
        return new HdNoorRaySphereLight(SdfPath());
    if (typeId == HdPrimTypeTokens->rectLight)
        return new HdNoorRayRectLight(SdfPath());
    if (typeId == HdPrimTypeTokens->diskLight)
        return new HdNoorRayDiskLight(SdfPath());
    if (typeId == HdPrimTypeTokens->distantLight)
        return new HdNoorRayDistantLight(SdfPath());
    if (typeId == HdPrimTypeTokens->domeLight)
        return new HdNoorRayDomeLight(SdfPath());
    return nullptr;
}

void HdNoorRayRenderDelegate::DestroySprim(HdSprim* sprim)
{
    delete sprim;
}

HdBprim* HdNoorRayRenderDelegate::CreateBprim(
    const TfToken& typeId, const SdfPath& id)
{
    if (typeId == HdPrimTypeTokens->renderBuffer)
        return new HdNoorRayRenderBuffer(id);
    return nullptr;
}

HdBprim* HdNoorRayRenderDelegate::CreateFallbackBprim(const TfToken& typeId)
{
    if (typeId == HdPrimTypeTokens->renderBuffer)
        return new HdNoorRayRenderBuffer(SdfPath());
    return nullptr;
}

void HdNoorRayRenderDelegate::DestroyBprim(HdBprim* bprim)
{
    delete bprim;
}

void HdNoorRayRenderDelegate::SetRenderSetting(
    const TfToken& key, const VtValue& value)
{
    if (!MaterialXMaterialLeaf(key).empty()) {
        // These dynamic settings are a high-volume transport channel rather
        // than renderer UI settings. Keep only the immutable snapshot in the
        // render param instead of retaining another XML copy in _settingsMap.
        if (StoreMaterialXSetting(*renderParam_, key, value))
            ++_settingsVersion;
        return;
    }
    HdRenderDelegate::SetRenderSetting(key, value);
}

void HdNoorRayRenderDelegate::CommitResources(HdChangeTracker*)
{
    resourceRegistry_->Commit();
    renderParam_->PruneTextureCache();
}

HdRenderSettingDescriptorList
HdNoorRayRenderDelegate::GetRenderSettingDescriptors() const
{
    return {
        {"Samples", TfToken("samples"), VtValue(64)},
        {"AOVs During Camera Motion", TfToken("aovEnabled"), VtValue(1)},
        {"Maximum Bounces", TfToken("maxBounces"), VtValue(8)},
        {"Indirect Light Clamp", TfToken("indirectLightClamp"), VtValue(10.0f)},
        {"Transparent Background", TfToken("transparentBackground"), VtValue(0)},
        {"Gaussian Cutoff Sigma", TfToken("gaussianCutoffSigma"), VtValue(3.0f)},
        {"Gaussian Proxy Type", TfToken("gaussianProxyType"), VtValue(3)},
        {"Gaussian Shading Mode", TfToken("gaussianShadingMode"), VtValue(1)},
        {"Gaussian SH Degree", TfToken("gaussianRenderSphericalHarmonics"), VtValue(3)},
        {"Buffer View", TfToken("bufferVisualization"), VtValue(0)},
        {"Gaussian Proxy Overdraw", TfToken("gaussianProxyOverdrawVisualization"), VtValue(0)},
        {"Gaussian Proxy Overdraw Range", TfToken("gaussianProxyOverdrawMax"), VtValue(1024)},
    };
}

HdAovDescriptor HdNoorRayRenderDelegate::GetDefaultAovDescriptor(
    const TfToken& name) const
{
    if (name == HdAovTokens->color)
        return {
            HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.0f, 0.0f, 0.0f, 1.0f))};
    return {};
}

VtDictionary HdNoorRayRenderDelegate::GetRenderStats() const
{
    return {
        {"rendererName", VtValue(std::string("NoorRay"))},
        {"percentDone", VtValue(renderParam_->GetProgress() * 100.0)},
        {"totalClockTime", VtValue(renderParam_->GetTotalClockTime())},
    };
}

TfTokenVector HdNoorRayRenderDelegate::GetMaterialRenderContexts() const
{
    return {TfToken("mtlx"), TfToken()};
}

bool HdNoorRayRenderDelegate::IsParallelSyncEnabled(
    const TfToken& primType) const
{
    // Mesh conversion operates on Hydra-owned inputs and local buffers
    // outside the render-param lock; scene registry mutations are serialized.
    // Material Sync is independently bounded in the render param so parallel
    // Hio work cannot create an unbounded decoded-image memory wave.
    return primType == HdPrimTypeTokens->mesh
        || primType == HdPrimTypeTokens->material;
}

PXR_NAMESPACE_CLOSE_SCOPE
