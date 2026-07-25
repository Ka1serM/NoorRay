#pragma once

#include "../api.h"

#include <pxr/imaging/hd/light.h>

#include <cstdint>
#include <string>

#include "Scene/Handle.h"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

class LightInstance;

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayLight : public HdLight
{
public:
    HdDirtyBits GetInitialDirtyBitsMask() const override;

protected:
    explicit HdNoorRayLight(const SdfPath& id);

    static float FloatParam(
        HdSceneDelegate*, const SdfPath&, const TfToken&, float fallback);
    static bool BoolParam(
        HdSceneDelegate*, const SdfPath&, const TfToken&, bool fallback);
    static glm::vec3 ColorParam(
        HdSceneDelegate*, const SdfPath&, const TfToken&,
        const glm::vec3& fallback);
    static std::string AssetPathParam(
        HdSceneDelegate*, const SdfPath&, const TfToken&);
    static glm::vec3 LightColor(HdSceneDelegate*, const SdfPath&);
    static float LightIntensity(HdSceneDelegate*, const SdfPath&);
    static glm::mat4 ToGlm(const GfMatrix4d&);
    static float BasisScale(const glm::mat4&, int column);
};

class HDNOORRAY_API HdNoorRayAnalyticLight : public HdNoorRayLight
{
public:
    void Sync(
        HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) final;
    void Finalize(HdRenderParam*) final;

protected:
    explicit HdNoorRayAnalyticLight(const SdfPath& id);

    virtual int NoorRayLightType(HdSceneDelegate*) const = 0;
    virtual void Configure(
        HdSceneDelegate*, const glm::mat4&, LightInstance&,
        float& intensity) const = 0;

private:
    SceneObjectHandle object_;
    int lightType_{-1};
};

PXR_NAMESPACE_CLOSE_SCOPE
