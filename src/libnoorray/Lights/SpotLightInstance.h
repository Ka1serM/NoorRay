#pragma once

#include <noorrhi/noorrhi.hpp>

#include "Lights/LightInstance.h"

class SpotLightInstance final
    : public LightInstance
    , public noorrhi::Shared<nr::graphics::SpotLight> {
public:
    SpotLightInstance(Scene& scene, const std::string& name,
                      const Transform& transform);
    std::unique_ptr<SceneObject> clone() const override;
    glm::vec3 getColor() const override { return data.color; }
    void setColor(const glm::vec3& color) override;
    float getIntensity() const override { return data.intensity; }
    void setIntensity(float intensity) override;
    float getSoftRadius() const override { return data.softRadius; }
    void setSoftRadius(float radius) override;
    void setSpotRadius(float radius) { setSoftRadius(radius); }
    void setSpotAngles(float innerDegrees, float outerDegrees);

protected:
    void updateTransformData(const Transform& world) override;
};
