#pragma once

#include "Scene/SceneObjectVisitor.h"

class ObjectUiVisitor final : public SceneObjectVisitor
{
public:
    void visit(SceneObject& object) override;
    void visit(MeshInstance& instance) override;
    void visit(GaussianInstance& instance) override;
    void visit(LightInstance& instance) override;
    void visit(CameraInstance& instance) override;

    bool changed{};
};

namespace domain_ui
{
bool render(SceneObject& object);
}
