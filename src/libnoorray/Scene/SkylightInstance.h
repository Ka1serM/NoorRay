#pragma once

#include <memory>
#include <string>

#include "Scene/SceneObject.h"

// The scene's sky light as an object, so it is listed and selected like a
// light. The scene has one Environment; its settings stay there and this
// object only gives them a place in the outliner and inspector.
class SkylightInstance final : public SceneObject {
public:
    SkylightInstance(Scene& scene, const std::string& name)
        : SceneObject(scene, name, Transform{}) {}

    std::unique_ptr<SceneObject> clone() const override {
        return std::make_unique<SkylightInstance>(*this);
    }
};
