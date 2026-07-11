#pragma once

#include <variant>

#include "Scene/SceneObject.h"
#include "Light/DirectionalLight.h"
#include "Light/PointLight.h"
#include "Light/SpotLight.h"
#include "Light/RectLight.h"

class LightInstance : public SceneObject {
public:
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
    std::string getType() const override;
    bool renderUi() override;
    void onTransformUpdated() override;
    glm::vec3 getColor() const;

private:
    friend class Scene;
    std::variant<PointLight, SpotLight, RectLight, DirectionalLight> light;
};
