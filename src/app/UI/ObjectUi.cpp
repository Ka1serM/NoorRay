#include "UI/ObjectUi.h"

#include "Camera/CameraInstance.h"
#include "Scene/GaussianInstance.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

bool domain_ui::render(SceneObject& object)
{
    bool changed = object_ui::render(object);
    if (auto* instance = dynamic_cast<MeshInstance*>(&object))
        changed |= object_ui::render(*instance);
    else if (auto* instance = dynamic_cast<GaussianInstance*>(&object))
        changed |= object_ui::render(*instance);
    else if (auto* instance = dynamic_cast<LightInstance*>(&object))
        changed |= object_ui::render(*instance);
    else if (auto* instance = dynamic_cast<CameraInstance*>(&object))
        changed |= object_ui::render(*instance);
    return changed;
}
