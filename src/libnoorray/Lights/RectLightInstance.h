#pragma once

#include <noorrhi/noorrhi.hpp>

#include "Lights/LightInstance.h"

class RectLightInstance final
    : public LightInstance
    , public noorrhi::Shared<nr::graphics::RectLight> {
public:
    RectLightInstance(Scene& scene, const std::string& name,
                      const Transform& transform);
    std::unique_ptr<SceneObject> clone() const override;
    glm::vec3 getColor() const override { return data.color; }
    void setColor(const glm::vec3& color) override;
    float getIntensity() const override { return data.intensity; }
    void setIntensity(float intensity) override;
    float getSoftRadius() const override { return 0.0f; }
    void setSoftRadius(float) override {}
    void setRectSize(float width, float height);

protected:
    void updateTransformData(const Transform& world) override;
};
