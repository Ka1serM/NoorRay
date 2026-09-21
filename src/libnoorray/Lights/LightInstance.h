#pragma once

#include <memory>
#include <string>

#include <glm/vec3.hpp>

#include <noorrhi/noorrhi.hpp>

#include "Scene/SceneObject.h"
#include "Shared/Light.h"

class Scene;

// The scene owns one typed CPU/GPU record per light family. Keeping the
// payload type in the C++ class also keeps the corresponding Scene vectors
// and NoorRHI structured buffers type-safe.
class LightInstance : public SceneObject {
public:
    static constexpr int TypePoint = 0;
    static constexpr int TypeSpot = 1;
    static constexpr int TypeRect = 2;
    static constexpr int TypeDirectional = 3;

    LightInstance(Scene& scene, const std::string& name,
                  const Transform& transform, int type);
    LightInstance(const LightInstance&) = delete;
    LightInstance& operator=(const LightInstance&) = delete;
    ~LightInstance() = default;

    virtual std::unique_ptr<SceneObject> clone() const override = 0;

    std::string getType() const;
    int getLightType() const { return lightType; }
    uint32_t getLightIndex() const { return lightIndex; }
    void setLightIndex(uint32_t index) { lightIndex = index; }

    void onTransformUpdated() override;

    virtual glm::vec3 getColor() const = 0;
    virtual void setColor(const glm::vec3& color) = 0;
    virtual float getIntensity() const = 0;
    virtual void setIntensity(float intensity) = 0;
    virtual float getSoftRadius() const = 0;
    virtual void setSoftRadius(float radius) = 0;
    void setPhotometry(const glm::vec3& color, float intensity);

    // Synchronize the typed record with the scene's typed light vector. GPU
    // buffer uploads remain the renderer's responsibility and happen outside
    // an active frame.
    void commitLightChanges();

protected:
    int lightType;
    uint32_t lightIndex{~0u};

    virtual void updateTransformData(const Transform& world) = 0;
    void updateSceneRecord();
    void copyCommonStateTo(LightInstance& target) const;
};
