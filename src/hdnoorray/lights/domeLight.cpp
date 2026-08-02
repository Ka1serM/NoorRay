#include "domeLight.h"

#include "../renderParam.h"

#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include <mutex>

#include <glm/gtc/matrix_transform.hpp>

#include "Scene/Resources/Environment.h"
#include "Scene/Scene.h"
#include "Scene/Resources/Texture.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayDomeLight::HdNoorRayDomeLight(const SdfPath& id)
    : HdNoorRayLight(id)
{
}

void HdNoorRayDomeLight::Sync(
    HdSceneDelegate* delegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    const glm::vec3 color = LightColor(delegate, GetId());
    const float exposure = delegate->GetVisible(GetId())
        ? LightIntensity(delegate, GetId()) : 0.0f;
    const std::string texturePath = AssetPathParam(
        delegate, GetId(), HdLightTokens->textureFile);
    const TextureRef texture = texturePath.empty()
        ? TextureRef{}
        // Material images use bottom-origin rows to match Blender's UV
        // buffers. Dome maps are sampled by the environment's own
        // equirectangular convention and must retain their source row order.
        : param.GetOrCreateTexture(texturePath, TextureEncoding::Srgb8, false);

    std::scoped_lock lock(param.mutex);
    Scene& scene = param.session.scene;
    scene.synchronizeBeforeMutation();
    Environment& environment = scene.getEnvironment();

    environment.color = color;
    environment.lightingExposure = exposure;
    environment.visibleExposure = 0.0f;
    environment.rotation = 0.0f;

    // Cycles evaluates Blender world directions as Z-up:
    // longitude = atan2(y, x), latitude = acos(z). NoorRay is Y-up and uses
    // atan2(z, x), so this fixed basis change preserves the texture's native
    // equirectangular projection without applying the USD dome transform.
    const glm::mat3 blenderWorldToNoorRay = glm::mat3(glm::rotate(
        glm::mat4(1.0f), glm::radians(-90.0f),
        glm::vec3(1.0f, 0.0f, 0.0f)));
    environment.setEquirectangularMapping(blenderWorldToNoorRay);

    if (!texture.isValid())
        scene.clearEnvironmentTexture();
    else
        scene.setEnvironmentTexture(texture);
    environment.updateDerivedSettings();
    active_ = true;
    scene.setDirtyFlag(EnvironmentCdf);
    scene.setDirtyFlag(Accumulation);
    *dirtyBits = Clean;
}

void HdNoorRayDomeLight::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    if (!active_)
        return;

    Scene& scene = param.session.scene;
    scene.synchronizeBeforeMutation();
    Environment& environment = scene.getEnvironment();
    scene.clearEnvironmentTexture();
    environment.color = glm::vec3(0.0f);
    environment.lightingExposure = 0.0f;
    environment.visibleExposure = 0.0f;
    environment.rotation = 0.0f;
    environment.setEquirectangularMapping();
    environment.updateDerivedSettings();
    scene.setDirtyFlag(EnvironmentCdf);
    scene.setDirtyFlag(Accumulation);
    active_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
