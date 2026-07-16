#include "Scene.h"
#include <algorithm>
#include "Camera/CameraInstance.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/GaussianInstance.h"
#include "Light/PointLight.h"
#include "Light/RectLight.h"
#include "Light/SpotLight.h"
#include "Light/DirectionalLight.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "Scene/SceneObject.h"
#include "Scene/SceneImporter.h"
#include "Scene/SceneReader.h"

Scene::Scene(Context& context)
    : context(context),
      environment(nr::rstd::make_unique<Environment>()),
      pointLights(context),
      spotLights(context),
      rectLights(context),
      directionalLights(context)
{
    auto camera = std::make_unique<PerspectiveCamera>();
    viewportCamera = std::make_shared<CameraInstance>(
        std::move(camera), "Viewport Camera", Transform(vec3(0.f, 0.f, 5.f)));
    viewportCamera->scene = this;
}

Scene::~Scene() = default;

void Scene::synchronizeBeforeMutation()
{
    if (mutationBarrier)
        mutationBarrier();
}

void Scene::load(const std::string& path)
{
    if (SceneImporter::IsSceneFile(path))
        read(path);
    else
    {
        clear();
        importFile(path);
    }
}

void Scene::importFile(const std::string& path)
{
    synchronizeBeforeMutation();
    SceneImporter::ImportFile(*this, path);
}

void Scene::read(const std::string& path)
{
    synchronizeBeforeMutation();
    SceneReader::Read(*this, path);
}

void Scene::notifyGeometryChanged() {
    setDirtyFlag(TLAS);
    setDirtyFlag(Accumulation);
}

// ── Internal registration ─────────────────────────────────────────────────────

uint32_t Scene::registerObject(std::unique_ptr<SceneObject> sceneObject) {
    sceneObject->scene = this;
    std::shared_ptr<SceneObject> sharedObject(std::move(sceneObject));
    sharedObject->setId(nextObjectId++);

    // Adding another camera must not unexpectedly change the rendered view.
    // The first camera is selected as a useful default; later changes are explicit.
    if (auto camera = std::dynamic_pointer_cast<CameraInstance>(sharedObject);
        camera && activeCamera.expired())
        activateCamera(camera);

    if (auto light = std::dynamic_pointer_cast<LightInstance>(sharedObject))
        registerLight(*light);
    const auto gaussianInstance = std::dynamic_pointer_cast<GaussianInstance>(sharedObject);
    if (gaussianInstance)
        setDirtyFlag(GaussianData);

    notifyGeometryChanged();
    sceneObjects.push_back(std::move(sharedObject));
    if (gaussianInstance)
    {
        gaussianInstance->sceneInstanceIndex = static_cast<uint32_t>(gaussianInstances.size());
        gaussianCount += gaussianInstance->getGaussianAsset().getGaussianCount();
        gaussianInstances.push_back(gaussianInstance);
        dirtyGaussianInstanceFlags.push_back(0);
    }
    return static_cast<uint32_t>(sceneObjects.size() - 1);
}

void Scene::rebuildGaussianInstanceCache()
{
    gaussianInstances.clear();
    gaussianCount = 0;
    for (const auto& object : sceneObjects)
    {
        if (auto instance = std::dynamic_pointer_cast<GaussianInstance>(object))
        {
            instance->sceneInstanceIndex = static_cast<uint32_t>(gaussianInstances.size());
            gaussianCount += instance->getGaussianAsset().getGaussianCount();
            gaussianInstances.push_back(std::move(instance));
        }
    }
    dirtyGaussianInstanceIndices.clear();
    dirtyGaussianInstanceFlags.assign(gaussianInstances.size(), 0);
}

// ── Public lifetime API ───────────────────────────────────────────────────────

void Scene::clear() {
    synchronizeBeforeMutation();
    // Switch rendering to the persistent viewport camera before scene-owned
    // cameras are destroyed.
    activateCamera(nullptr);
    sceneObjects.clear();
    gaussianInstances.clear();
    gaussianCount = 0;
    meshAssets.clear();
    gaussianAssets.clear();
    textures.clear();
    textureNames.clear();
    pointLights.clear();
    spotLights.clear();
    rectLights.clear();
    directionalLights.clear();
    dirtyMeshInstanceIndices.clear();
    dirtyGaussianInstanceIndices.clear();
    dirtyGaussianInstanceFlags.clear();
    gaussianOpacities.clear();
    gaussianShCoeffs.clear();
    copiedObject.reset();
    activeObjectId = 0;
    nextObjectId = 1;
    renderSettings = {};
    environment->destroyCdf();
    environment->textureIndex = -1;
    environment->color = vec3(1.0f);
    environment->rotation = 0.0f;
    environment->visible = 1;
    environment->visibleExposure = 0.0f;
    environment->lightingExposure = 1.0f;
    environment->setEquirectangularMapping();
    environment->updateDerivedSettings();
    dirtyFlags = TLAS | Meshes | Textures | EnvironmentCdf | Lights
        | CameraState | Accumulation | GaussianData;
    ++lightRevision;
}

uint64_t Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    synchronizeBeforeMutation();
    const uint32_t index = registerObject(std::move(sceneObject));
    return sceneObjects[index]->getId();
}

uint32_t Scene::add(MeshAsset meshAsset) {
    synchronizeBeforeMutation();
    const uint32_t index = static_cast<uint32_t>(meshAssets.size());
    meshAsset.setMeshIndex(index);
    meshAssets.push_back(std::move(meshAsset));
    setDirtyFlag(Meshes);
    return index;
}

uint32_t Scene::add(GaussianAsset gaussianAsset) {
    synchronizeBeforeMutation();
    const uint32_t index = static_cast<uint32_t>(gaussianAssets.size());
    gaussianAssets.push_back(std::move(gaussianAsset));
    setDirtyFlag(TLAS);
    setDirtyFlag(GaussianData);
    return index;
}

Texture& Scene::add(Texture&& texture) {
    synchronizeBeforeMutation();
    texture.sceneIndex = static_cast<int>(textures.size());
    textureNames.push_back(texture.getName());
    textures.push_back(std::move(texture));
    setDirtyFlag(Textures);
    return textures.back();
}

bool Scene::remove(SceneObject* objToRemove) {
    if (!objToRemove)
        return false;

    // Recursively remove children first
    while (true) {
        const auto children = objToRemove->getChildren();
        if (children.empty()) break;
        remove(children.back().get());
    }

    if (objToRemove->getParent())
        objToRemove->getParent()->removeChild(objToRemove);

    if (auto* light = dynamic_cast<LightInstance*>(objToRemove))
        unregisterLight(*light);
    if (dynamic_cast<GaussianInstance*>(objToRemove))
        setDirtyFlag(GaussianData);

    const auto it = std::ranges::find_if(sceneObjects, [objToRemove](const auto& ptr) {
        return ptr.get() == objToRemove;
    });
    if (it == sceneObjects.end())
        return false;

    if (activeObjectId == (*it)->getId())
        activeObjectId = 0;
    if (objToRemove == getActiveCamera()) {
        const auto replacement = std::ranges::find_if(sceneObjects,
            [objToRemove](const std::shared_ptr<SceneObject>& object) {
                return object.get() != objToRemove
                    && dynamic_cast<CameraInstance*>(object.get()) != nullptr;
            });
        activateCamera(replacement != sceneObjects.end()
            ? std::static_pointer_cast<CameraInstance>(*replacement) : nullptr);
    }

    objToRemove->scene = nullptr;
    sceneObjects.erase(it);
    rebuildGaussianInstanceCache();
    notifyGeometryChanged();
    return true;
}

bool Scene::removeObject(const uint64_t objectId) {
    synchronizeBeforeMutation();
    return remove(findObjectPtr(objectId).get());
}

bool Scene::replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject) {
    synchronizeBeforeMutation();
    if (!oldObject || !newObject)
        return false;

    const auto it = std::ranges::find_if(sceneObjects, [oldObject](const auto& ptr) {
        return ptr.get() == oldObject;
    });
    if (it == sceneObjects.end())
        return false;

    const uint32_t index = static_cast<uint32_t>(std::distance(sceneObjects.begin(), it));
    const bool wasActiveCamera = oldObject == getActiveCamera();
    const bool replacedCamera = dynamic_cast<CameraInstance*>(oldObject) != nullptr;
    const bool replacedGaussian = dynamic_cast<GaussianInstance*>(oldObject) != nullptr;
    const bool replacementGaussian = dynamic_cast<GaussianInstance*>(newObject.get()) != nullptr;
    SceneObject* parent = oldObject->getParent();
    const auto parentPtr = findObjectPtr(parent);
    const auto children = oldObject->getChildren();
    const bool wasCopied = copiedObject.lock().get() == oldObject;

    if (parent)
        parent->removeChild(oldObject);
    if (auto* oldLight = dynamic_cast<LightInstance*>(oldObject))
        unregisterLight(*oldLight);
    oldObject->clearParent();
    oldObject->children.clear();
    oldObject->scene = nullptr;

    std::shared_ptr<SceneObject> newShared(std::move(newObject));
    newShared->scene = this;
    newShared->setId(oldObject->getId());

    if (auto* newLight = dynamic_cast<LightInstance*>(newShared.get()))
        registerLight(*newLight);

    if (wasActiveCamera || activeCamera.expired())
        activateCamera(std::dynamic_pointer_cast<CameraInstance>(newShared));

    *it = std::move(newShared);
    rebuildGaussianInstanceCache();

    if (replacedCamera || dynamic_cast<CameraInstance*>(sceneObjects[index].get()))
        setDirtyFlag(CameraState);
    if (replacedGaussian || replacementGaussian)
        setDirtyFlag(GaussianData);

    if (parentPtr)
        parentPtr->addChild(sceneObjects[index]);
    for (const auto& child : children)
        sceneObjects[index]->addChild(child);
    if (wasCopied)
        copiedObject = sceneObjects[index];

    notifyGeometryChanged();
    return true;
}

void Scene::activateCamera(const std::shared_ptr<CameraInstance>& camera)
{
    const std::shared_ptr<CameraInstance> previous = activeCamera.lock();
    if (previous == camera)
        return;

    if (previous)
        previous->setArcballActive(false);
    if (camera)
        camera->setArcballActive(false);
    activeCamera = camera;
    ++activeCameraRevision;
    setDirtyFlag(CameraState);
    setDirtyFlag(Accumulation);
}

bool Scene::setActiveCamera(CameraInstance* camera) {
    if (!camera) {
        activateCamera(nullptr);
        return true;
    }

    const auto object = findObjectPtr(camera);
    const auto cameraPtr = std::dynamic_pointer_cast<CameraInstance>(object);
    if (!cameraPtr)
        return false;

    activateCamera(cameraPtr);
    return true;
}

// ── Light management ─────────────────────────────────────────────────────────

uint32_t Scene::registerLight(LightInstance& light)
{
    uint32_t idx = UINT32_MAX;
    switch (light.lightType) {
    case LightInstance::TypePoint:
        idx = static_cast<uint32_t>(pointLights.size());
        pointLights.push_back(std::get<PointLight>(light.light));
        break;
    case LightInstance::TypeSpot:
        idx = static_cast<uint32_t>(spotLights.size());
        spotLights.push_back(std::get<SpotLight>(light.light));
        break;
    case LightInstance::TypeRect:
        idx = static_cast<uint32_t>(rectLights.size());
        rectLights.push_back(std::get<RectLight>(light.light));
        break;
    case LightInstance::TypeDirectional:
        idx = static_cast<uint32_t>(directionalLights.size());
        directionalLights.push_back(std::get<DirectionalLight>(light.light));
        break;
    }
    light.lightIndex = idx;
    setDirtyFlag(Lights);
    setDirtyFlag(Accumulation);
    return idx;
}

void Scene::unregisterLight(LightInstance& light)
{
    const uint32_t idx = light.lightIndex;
    switch (light.lightType) {
    case LightInstance::TypePoint:
        { const uint32_t displaced = static_cast<uint32_t>(pointLights.size() - 1);
        if (idx != displaced) pointLights[idx] = std::move(pointLights.back());
        pointLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypePoint && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    case LightInstance::TypeSpot:
        { const uint32_t displaced = static_cast<uint32_t>(spotLights.size() - 1);
        if (idx != displaced) spotLights[idx] = std::move(spotLights.back());
        spotLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypeSpot && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    case LightInstance::TypeRect:
        { const uint32_t displaced = static_cast<uint32_t>(rectLights.size() - 1);
        if (idx != displaced) rectLights[idx] = std::move(rectLights.back());
        rectLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypeRect && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    case LightInstance::TypeDirectional:
        { const uint32_t displaced = static_cast<uint32_t>(directionalLights.size() - 1);
        if (idx != displaced) directionalLights[idx] = std::move(directionalLights.back());
        directionalLights.pop_back();
        for (auto& obj : sceneObjects)
            if (auto* li = dynamic_cast<LightInstance*>(obj.get()))
                if (li->lightType == LightInstance::TypeDirectional && li->lightIndex == displaced)
                    { li->lightIndex = idx; break; }
        }
        break;
    }
    light.lightIndex = UINT32_MAX;
    setDirtyFlag(Lights);
    setDirtyFlag(Accumulation);
}

// ── Hierarchy ────────────────────────────────────────────────────────────────

void Scene::reparent(SceneObject* objectToMove, SceneObject* newParent) {
    if (!objectToMove || objectToMove == newParent)
        return;

    const auto objectToMovePtr = findObjectPtr(objectToMove);
    const auto newParentPtr = findObjectPtr(newParent);
    if (!objectToMovePtr || (newParent && !newParentPtr))
        return;

    for (SceneObject* p = newParent; p != nullptr; p = p->getParent())
        if (p == objectToMove)
            return;

    const mat4 oldWorldMatrix = objectToMove->getWorldTransform().getMatrix();

    if (objectToMove->getParent())
        objectToMove->getParent()->removeChild(objectToMove);

    if (newParentPtr) {
        newParentPtr->addChild(objectToMovePtr);
        const mat4 newLocal = inverse(newParent->getWorldTransform().getMatrix()) * oldWorldMatrix;
        objectToMove->setLocalTransform(Transform{newLocal});
    } else {
        objectToMove->clearParent();
        objectToMove->setLocalTransform(Transform{oldWorldMatrix});
    }
}

bool Scene::reparentObject(const uint64_t objectId, const uint64_t newParentId) {
    synchronizeBeforeMutation();
    const auto objectToMove = findObjectPtr(objectId);
    const auto newParent = newParentId != 0 ? findObjectPtr(newParentId) : nullptr;
    if (!objectToMove || (newParentId != 0 && !newParent))
        return false;

    reparent(objectToMove.get(), newParent.get());
    return true;
}

// ── Clipboard ────────────────────────────────────────────────────────────────

void Scene::copyObject(const uint64_t objectId) {
    copiedObject = findObjectPtr(objectId);
}

std::shared_ptr<SceneObject> Scene::cloneHierarchy(const SceneObject* source) {
    const uint32_t index = registerObject(source->clone());
    const auto newObject = sceneObjects[index];
    for (const auto& child : source->getChildren())
        if (child)
            newObject->addChild(cloneHierarchy(child.get()));
    return newObject;
}

void Scene::paste() {
    synchronizeBeforeMutation();
    const auto source = copiedObject.lock();
    if (!source)
        return;

    const auto newObject = cloneHierarchy(source.get());

    SceneObject* targetParent = nullptr;
    if (SceneObject* active = getActiveObject())
        targetParent = active->getParent();

    reparent(newObject.get(), targetParent);
    setActiveObjectId(newObject->getId());
}

// ── Queries ───────────────────────────────────────────────────────────────────

std::vector<std::shared_ptr<SceneObject>> Scene::getRootObjects() const {
    std::vector<std::shared_ptr<SceneObject>> result;
    for (const auto& obj : sceneObjects)
        if (obj->getParent() == nullptr)
            result.push_back(obj);
    return result;
}

std::vector<std::shared_ptr<MeshInstance>> Scene::getMeshInstances() const {
    std::vector<std::shared_ptr<MeshInstance>> result;
    for (const auto& obj : sceneObjects)
        if (auto mi = std::dynamic_pointer_cast<MeshInstance>(obj))
            result.push_back(mi);
    return result;
}

void Scene::markMeshInstanceTransformDirty(const uint32_t instanceIndex) {
    if (instanceIndex == ~0u)
        return;
    if (std::ranges::find(dirtyMeshInstanceIndices, instanceIndex) == dirtyMeshInstanceIndices.end())
        dirtyMeshInstanceIndices.push_back(instanceIndex);
}

uint32_t Scene::getMeshInstanceIndex(const SceneObject* object) const {
    uint32_t instanceIndex = 0;
    for (const auto& obj : sceneObjects) {
        if (!std::dynamic_pointer_cast<MeshInstance>(obj))
            continue;
        if (obj.get() == object)
            return instanceIndex;
        ++instanceIndex;
    }
    return ~0u;
}

void Scene::markGaussianInstanceTransformDirty(const uint32_t instanceIndex)
{
    if (instanceIndex >= dirtyGaussianInstanceFlags.size()
        || dirtyGaussianInstanceFlags[instanceIndex])
        return;
    dirtyGaussianInstanceFlags[instanceIndex] = 1;
    dirtyGaussianInstanceIndices.push_back(instanceIndex);
}

void Scene::buildGaussianRenderData()
{
    struct GaussianSpan
    {
        const Gaussian* gaussians;
        uint32_t offset;
    };

    std::vector<GaussianSpan> spans;
    uint32_t total = 0;
    for (const auto& instance : getGaussianInstances())
    {
        const auto& gaussians = instance->getGaussianAsset().getGaussians();
        spans.push_back({gaussians.data(), total});
        total += static_cast<uint32_t>(gaussians.size());
    }

    gaussianOpacities.resize(total);
    constexpr uint32_t coefficientsPerGaussian = 16;
    gaussianShCoeffs.resize(static_cast<size_t>(total) * coefficientsPerGaussian);
    if (total == 0)
        return;

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, total, 1024),
        [&](const tbb::blocked_range<uint32_t>& range)
    {
        for (uint32_t globalIndex = range.begin(); globalIndex != range.end(); ++globalIndex)
        {
            const GaussianSpan* span = &spans.front();
            if (spans.size() > 1)
            {
                auto found = std::upper_bound(spans.begin(), spans.end(), globalIndex,
                    [](const uint32_t value, const GaussianSpan& candidate) {
                        return value < candidate.offset;
                    });
                span = &*--found;
            }
            const Gaussian& gaussian = span->gaussians[globalIndex - span->offset];

            gaussianOpacities[globalIndex] = gaussian.opacity;
            glm::vec3* coefficients = gaussianShCoeffs.data()
                + static_cast<size_t>(globalIndex) * coefficientsPerGaussian;
            for (uint32_t coefficient = 0; coefficient < coefficientsPerGaussian; ++coefficient)
                coefficients[coefficient] = glm::vec3(0.0f);
            const uint32_t count = std::min(gaussian.shCoeffCount, coefficientsPerGaussian);
            for (uint32_t coefficient = 0; coefficient < count; ++coefficient)
                coefficients[coefficient] = gaussian.shCoeffs[coefficient];
        }
    });
}

uint32_t Scene::getActiveMeshInstanceIndex() const
{
    uint32_t instanceIndex = 0;
    for (const auto& object : sceneObjects)
    {
        if (!std::dynamic_pointer_cast<MeshInstance>(object))
            continue;
        if (object->getId() == activeObjectId)
            return instanceIndex;
        ++instanceIndex;
    }
    return ~0u;
}

MeshAsset* Scene::getMeshAsset(const std::string& name) {
    for (auto& asset : meshAssets)
        if (asset.getPath() == name)
            return &asset;
    return nullptr;
}

const MeshAsset* Scene::getMeshAsset(const std::string& name) const {
    for (const auto& asset : meshAssets)
        if (asset.getPath() == name)
            return &asset;
    return nullptr;
}

// ── Lookups ───────────────────────────────────────────────────────────────────

std::shared_ptr<SceneObject> Scene::findObjectPtr(const SceneObject* object) const {
    if (!object) return nullptr;
    const auto it = std::ranges::find_if(sceneObjects, [object](const auto& ptr) {
        return ptr.get() == object;
    });
    return it != sceneObjects.end() ? *it : nullptr;
}

std::shared_ptr<SceneObject> Scene::findObjectPtr(const uint64_t objectId) const {
    if (objectId == 0) return nullptr;
    const auto it = std::ranges::find_if(sceneObjects, [objectId](const auto& ptr) {
        return ptr->getId() == objectId;
    });
    return it != sceneObjects.end() ? *it : nullptr;
}
