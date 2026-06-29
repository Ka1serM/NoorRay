#include "LightInstance.h"

#include <variant>

#include "Scene.h"

LightInstance::LightInstance(Scene& scene, const std::string& name,
                             const Transform& transform, int type)
    : SceneObject(scene, name, transform), lightType(type)
{
    switch (type) {
    case TypePoint: light = PointLight{}; break;
    case TypeSpot:  light = SpotLight{};  break;
    case TypeRect:  light = RectLight{};  break;
    case TypeDirectional: light = DirectionalLight{}; break;
    }
    onTransformUpdated();
}

void LightInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    const Transform world = getWorldTransform();
    const vec3 pos = world.getPosition();
    const quat rotation = world.getRotation();
    const vec3 dir = glm::normalize(rotation * vec3(0.f, 0.f, -1.f));
    const vec3 tangent = glm::normalize(rotation * vec3(1.f, 0.f, 0.f));

    switch (lightType) {
    case TypePoint: std::get<PointLight>(light).position = pos; break;
    case TypeSpot: { auto& l = std::get<SpotLight>(light); l.position = pos; l.direction = dir; break; }
    case TypeRect: { auto& l = std::get<RectLight>(light); l.position = pos; l.direction = dir; l.tangent = tangent; break; }
    case TypeDirectional: std::get<DirectionalLight>(light).direction = dir; break;
    }

    if (lightIndex == ~0u) return;

    switch (lightType) {
    case TypePoint:
        scene.pointLights[lightIndex].position = pos;
        break;
    case TypeSpot:
        scene.spotLights[lightIndex].position = pos;
        scene.spotLights[lightIndex].direction = dir;
        break;
    case TypeRect:
        scene.rectLights[lightIndex].position = pos;
        scene.rectLights[lightIndex].direction = dir;
        scene.rectLights[lightIndex].tangent = tangent;
        break;
    case TypeDirectional:
        scene.directionalLights[lightIndex].direction = dir;
        break;
    }
}

std::unique_ptr<SceneObject> LightInstance::clone() const
{
    auto c = std::make_unique<LightInstance>(scene, getName() + " (copy)",
                                             transform, lightType);
    c->light = light;
    return c;
}

std::string LightInstance::getType() const
{
    switch (lightType) {
    case TypePoint: return "Point Light";
    case TypeSpot:  return "Spot Light";
    case TypeRect:  return "Rect Light";
    case TypeDirectional: return "Directional Light";
    }
    return "Light";
}

bool LightInstance::renderUi()
{
    bool changed = SceneObject::renderUi();
    switch (lightType) {
    case TypePoint: changed |= scene.pointLights[lightIndex].renderUi(); break;
    case TypeSpot:  changed |= scene.spotLights[lightIndex].renderUi(); break;
    case TypeRect:  changed |= scene.rectLights[lightIndex].renderUi(); break;
    case TypeDirectional: changed |= scene.directionalLights[lightIndex].renderUi(); break;
    }
    if (changed)
        scene.setDirtyFlag(Accumulation);
    return changed;
}
