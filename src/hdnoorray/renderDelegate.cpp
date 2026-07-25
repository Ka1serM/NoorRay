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

namespace
{
// Render settings arrive from Python, where the value type is whatever the host
// happened to pass. Reading them defensively keeps a mistyped setting from
// throwing out of a Hydra callback and taking the host down with it.
int IntSetting(const VtValue& value, const int fallback)
{
    if (value.IsHolding<int>())
        return value.UncheckedGet<int>();
    if (value.IsHolding<bool>())
        return value.UncheckedGet<bool>() ? 1 : 0;
    if (value.IsHolding<float>())
        return static_cast<int>(value.UncheckedGet<float>());
    if (value.IsHolding<double>())
        return static_cast<int>(value.UncheckedGet<double>());
    return fallback;
}

float FloatSetting(const VtValue& value, const float fallback)
{
    if (value.IsHolding<float>())
        return value.UncheckedGet<float>();
    if (value.IsHolding<double>())
        return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<int>())
        return static_cast<float>(value.UncheckedGet<int>());
    return fallback;
}

std::string StringSetting(const VtValue& value, const std::string& fallback)
{
    if (value.IsHolding<std::string>())
        return value.UncheckedGet<std::string>();
    return fallback;
}
}

void HdNoorRayRenderDelegate::SetRenderSetting(
    const TfToken& key, const VtValue& value)
{
    // Intercept camera-specific settings and store on renderParam directly.
    CameraSettings cs = renderParam_->cameraSettings;
    bool isCamera = true;
    if (key == TfToken("cameraProjection"))
        cs.projectionType = IntSetting(value, cs.projectionType);
    else if (key == TfToken("cameraApertureDiameter"))
        cs.apertureDiameterMm = FloatSetting(value, cs.apertureDiameterMm);
    else if (key == TfToken("cameraBokehBias"))
        cs.bokehBias = FloatSetting(value, cs.bokehBias);
    else if (key == TfToken("cameraLensPath"))
        cs.lensPath = StringSetting(value, cs.lensPath);
    else if (key == TfToken("cameraGlassCatalogs"))
        cs.glassCatalogs = StringSetting(value, cs.glassCatalogs);
    else if (key == TfToken("cameraRayLutPath"))
        cs.rayLutPath = StringSetting(value, cs.rayLutPath);
    else if (key == TfToken("cameraSensorType"))
        cs.sensorType = IntSetting(value, cs.sensorType);
    else if (key == TfToken("cameraSensorPath"))
        cs.sensorPath = StringSetting(value, cs.sensorPath);
    else if (key == TfToken("cameraPsfPath"))
        cs.psfPath = StringSetting(value, cs.psfPath);
    else
        isCamera = false;

    if (isCamera) {
        if (renderParam_->cameraSettings != cs) {
            renderParam_->cameraSettings = cs;
            renderParam_->MarkRenderSettingsChanged();
        }
        return;
    }

    // Regular render setting
    if (GetRenderSetting(key) == value)
        return;
    HdRenderDelegate::SetRenderSetting(key, value);
    renderParam_->MarkRenderSettingsChanged();
}

void HdNoorRayRenderDelegate::CommitResources(HdChangeTracker*)
{
    resourceRegistry_->Commit();
}

HdRenderSettingDescriptorList
HdNoorRayRenderDelegate::GetRenderSettingDescriptors() const
{
    return {
        {"Samples", TfToken("samples"), VtValue(64)},
        {"Maximum Bounces", TfToken("maxBounces"), VtValue(8)},
        {"Noise Limit", TfToken("noiseLimitEnabled"), VtValue(0)},
        {"Noise Threshold", TfToken("noiseLevel"), VtValue(0.0001f)},
        {"OptiX Denoiser", TfToken("optixDenoiserEnabled"), VtValue(0)},
        {"Denoiser Min Samples", TfToken("optixDenoiserMinSamples"), VtValue(1)},
        {"Russian Roulette Start", TfToken("russianRouletteStartBounce"), VtValue(3)},
        {"Transparent Background", TfToken("transparentBackground"), VtValue(0)},
        {"Gaussian Cutoff Sigma", TfToken("gaussianCutoffSigma"), VtValue(3.0f)},
        {"Gaussian Proxy Type", TfToken("gaussianProxyType"), VtValue(3)},
        {"Gaussian Shading Mode", TfToken("gaussianShadingMode"), VtValue(1)},
        {"Gaussian SH Degree", TfToken("gaussianRenderSphericalHarmonics"), VtValue(3)},
        {"Gaussian Proxy Overdraw", TfToken("gaussianProxyOverdrawVisualization"), VtValue(0)},
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

PXR_NAMESPACE_CLOSE_SCOPE
