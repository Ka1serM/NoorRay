#pragma once

#include <noorrhi/noorrhi.hpp>

#include "Lights/LightInstance.h"

class PointLightInstance final
    : public LightInstance
    , public noorrhi::Shared<nr::graphics::PointLight> {
public:
    PointLightInstance(Scene& scene, const std::string& name,
                       const Transform& transform);
    std::unique_ptr<SceneObject> clone() const override;
    glm::vec3 getColor() const override { return data.color; }
    void setColor(const glm::vec3& color) override;
    float getIntensity() const override { return data.intensity; }
    void setIntensity(float intensity) override;
    float getSoftRadius() const override { return data.softRadius; }
    void setSoftRadius(float radius) override;
    void setPointRadius(float radius) { setSoftRadius(radius); }

protected:
    void updateTransformData(const Transform& world) override;
};
