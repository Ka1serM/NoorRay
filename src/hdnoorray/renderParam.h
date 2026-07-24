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
    uint64_t GetSceneVersion() const;

    int GetOrCreateTexture(
        const std::string& filePath, TextureEncoding encoding);
    void PublishMaterial(const SdfPath& id, const Material& material);
    void BindMaterial(const SdfPath& id, uint32_t meshIndex);
    void UnbindMaterial(const SdfPath& id, uint32_t meshIndex);
    CameraSettings cameraSettings;

    mutable std::mutex mutex;
    noorray::NoorRaySession session;

private:
    std::atomic<uint64_t> sceneVersion_{1};
    std::atomic<bool> renderSettingsChanged_{true};
    std::map<SdfPath, uint32_t> materials_;
    std::map<SdfPath, std::vector<uint32_t>> materialBindings_;
    std::map<std::string, int> textureCache_;
};

PXR_NAMESPACE_CLOSE_SCOPE
