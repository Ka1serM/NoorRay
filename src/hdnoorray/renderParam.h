#pragma once

#include "api.h"

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/usd/sdf/path.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

struct CameraSettings
{
    int projectionType = -1;
    float apertureDiameterMm = -1.0f;
    float bokehBias = -1.0f;
    std::string lensPath;
    std::string glassCatalogs;
    std::string rayLutPath;
    int sensorType = 0;
    std::string sensorPath;
    std::string psfPath;

    bool operator!=(const CameraSettings& o) const
    {
        return projectionType != o.projectionType
            || apertureDiameterMm != o.apertureDiameterMm
            || bokehBias != o.bokehBias
            || lensPath != o.lensPath
            || glassCatalogs != o.glassCatalogs
            || rayLutPath != o.rayLutPath
            || sensorType != o.sensorType
            || sensorPath != o.sensorPath
            || psfPath != o.psfPath;
    }
};

PXR_NAMESPACE_CLOSE_SCOPE

#include "NoorRaySession.h"
#include "Scene/MaterialRegistry.h"
#include "Scene/SceneResources.h"
#include "Scene/Texture.h"
#include "Shading/Material.h"

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayRenderParam final : public HdRenderParam
{
public:
    HdNoorRayRenderParam();
    ~HdNoorRayRenderParam() override;

    HdNoorRayRenderParam(const HdNoorRayRenderParam&) = delete;
    HdNoorRayRenderParam& operator=(const HdNoorRayRenderParam&) = delete;

    void MarkSceneDirty();
    void MarkRenderSettingsChanged();
    bool ConsumeRenderSettingsChanged();
    // Returns the scene version and bumps it if a sync is pending, so that
    // multiple MarkSceneDirty() calls within a single Hydra commit produce
    // exactly one scene version bump visible to the render loop.
    uint64_t GetSceneVersion();
    void SetProgress(double progress);
    double GetProgress() const;
    void AccumulateGpuTimeMs(float ms);
    double GetTotalClockTime() const;
    void ResetClock();

    // Textures are shared between the materials that name the same file. The
    // returned reference owns its share; dropping every one of them frees the
    // texture. Returns an empty reference when the file cannot be loaded.
    TextureRef GetOrCreateTexture(
        const std::string& filePath, TextureEncoding encoding);
    void PublishMaterial(const SdfPath& id, const Material& material);
    void ReleaseMaterial(const SdfPath& id);
    // Binds the material a mesh prim asked for. Meshes are addressed by handle
    // because the asset storage is a managed vector that moves its elements
    // when it grows.
    void BindMaterial(const SdfPath& id, MeshAssetHandle mesh);
    void UnbindMaterial(const SdfPath& id, MeshAssetHandle mesh);

    CameraSettings cameraSettings;

    mutable std::mutex mutex;
    noorray::NoorRaySession session;

private:
    void PruneTextureCache();

    std::atomic<uint64_t> sceneVersion_{1};
    std::atomic<bool> sceneNeedsUpdate_{false};
    std::atomic<bool> renderSettingsChanged_{true};
    std::atomic<double> progress_{};
    std::atomic<double> cumulativeGpuTimeSeconds_{};
    std::map<SdfPath, MaterialRef> materials_;
    std::map<SdfPath, std::vector<MeshAssetHandle>> materialBindings_;
    std::map<std::string, TextureRef> textureCache_;
};

PXR_NAMESPACE_CLOSE_SCOPE
