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
    void setSourceLength(float length);
    void setUnits(uint32_t units) override { data.units = units; updateSceneRecord(); }
    void setControls(const LightControls& controls) override { data.controls = controls; updateSceneRecord(); }
    void setFalloff(const LightFalloff& falloff) { data.falloff = falloff; updateSceneRecord(); }

protected:
    void updateTransformData(const Transform& world) override;
};
