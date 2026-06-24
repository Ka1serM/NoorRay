#pragma once

#include "CameraBase.h"

// Perspective camera with thin-lens depth-of-field.
// Mirrors GPU ThinLensCamera — aperture, focus distance, and bokeh bias are
// active parameters here; they are ignored by PerspectiveCamera (pinhole).
class ThinLensCamera final : public CameraBase {
public:
    ThinLensCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings = {});
    ThinLensCamera(const ThinLensCamera& other);

    std::unique_ptr<SceneObject> clone() const override;
    CameraProjectionType getProjectionType() const override { return CameraProjectionType::ThinLens; }
    const char* getProjectionName() const override { return "Thin Lens"; }

protected:
    void computeProjectionData(const vec3& direction, const vec3& up, const vec3& right, float aspectRatio) override;
};
