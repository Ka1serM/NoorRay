#pragma once

#include <variant>

#include "Scene/SceneObject.h"
#include "Light/DirectionalLight.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"

class LightInstance : public SceneObject {
public:
    using LightData = std::variant<PointLight, SpotLight, RectLight, DirectionalLight>;

    static constexpr int TypePoint = 0;
    static constexpr int TypeSpot  = 1;
    static constexpr int TypeRect  = 2;
    static constexpr int TypeDirectional = 3;

    int lightType{TypePoint};
    uint32_t lightIndex{~0u};

    LightInstance(Scene& scene, const std::string& name,
                  const Transform& transform, int type);
    LightInstance(const LightInstance&) = delete;
    LightInstance& operator=(const LightInstance&) = delete;

    std::unique_ptr<SceneObject> clone() const override;
    void accept(SceneObjectVisitor& visitor) override;
    std::string getType() const override;
    void onTransformUpdated() override;
    glm::vec3 getColor() const;
    void setPhotometry(const glm::vec3& color, float intensity);
    void setPointRadius(float radius);
    void setSpotRadius(float radius);
    void setSpotAngles(float innerDegrees, float outerDegrees);
    void setDirectionalSoftAngle(float degrees);
    LightData& getLightData() { return light; }
    const LightData& getLightData() const { return light; }
    void commitLightChanges();

private:
    friend class Scene;
    LightData light;
};
