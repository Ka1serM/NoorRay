#include "UI/ObjectUi.h"

#include "Scene/SceneObject.h"

bool domain_ui::render(SceneObject& object)
{
    ObjectUiVisitor visitor;
    object.accept(visitor);
    return visitor.changed;
}
