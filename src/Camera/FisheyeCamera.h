#pragma once

#include "CameraBase.h"

class FisheyeCamera final : public CameraBase {
public:
    FisheyeCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    FisheyeCamera(const FisheyeCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Fisheye; }
    const char* getProjectionName() const override { return "Fisheye"; }
    bool supportsDOF() const override { return true; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
};
