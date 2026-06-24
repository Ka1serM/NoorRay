#pragma once

#include "CameraBase.h"

class PerspectiveCamera final : public CameraBase {
public:
    PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    PerspectiveCamera(Scene& scene, const std::string& name, Transform transform, float aspect, float sensorWidth, float sensorHeight, float focalLength, float aperture, float focusDistance, float bokehBias);
    PerspectiveCamera(const PerspectiveCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Perspective; }
    const char* getProjectionName() const override { return "Perspective"; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
};
