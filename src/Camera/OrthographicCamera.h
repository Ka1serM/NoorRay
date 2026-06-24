#pragma once

#include "CameraBase.h"

class OrthographicCamera final : public CameraBase {
public:
    OrthographicCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    OrthographicCamera(const OrthographicCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::Orthographic; }
    const char* getProjectionName() const override { return "Orthographic"; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
};
