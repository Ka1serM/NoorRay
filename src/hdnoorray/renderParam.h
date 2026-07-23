#pragma once

#include "api.h"

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/usd/sdf/path.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "NoorRaySession.h"
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
    uint64_t GetSceneVersion() const;
    void PublishMaterial(const SdfPath& id, const Material& material);
    void BindMaterial(const SdfPath& id, uint32_t meshIndex);
    void UnbindMaterial(const SdfPath& id, uint32_t meshIndex);

    mutable std::mutex mutex;
    noorray::NoorRaySession session;

private:
    std::atomic<uint64_t> sceneVersion_{1};
    std::map<SdfPath, uint32_t> materials_;
    std::map<SdfPath, std::vector<uint32_t>> materialBindings_;
};

PXR_NAMESPACE_CLOSE_SCOPE
