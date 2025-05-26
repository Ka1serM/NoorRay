#pragma once

#include <string>
#include "Transform.h"

class SceneObject {

protected:

    Transform transform;

    void renderUiBegin();

    void renderUiEnd();

public:
    virtual ~SceneObject() = default;

    std::string name;

    SceneObject(const std::string &name, const Transform &transform);

    glm::mat4 getTransformMatrix() const;

    void renderUi();

    virtual void renderUiOverride();
};
