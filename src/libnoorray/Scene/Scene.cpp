#include "Scene.h"
#include <algorithm>
#include <optional>
#include <utility>
#include "Camera/CameraInstance.h"
#include "Scene/LightInstance.h"
#include "Lights/DirectionalLightInstance.h"
#include "Lights/PointLightInstance.h"
#include "Lights/RectLightInstance.h"
#include "Lights/SpotLightInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

using glm::inverse;
using glm::mat4;
using glm::vec3;

Scene::Scene()
{
    // Attach after all Scene members are initialized: environment setters
    // publish changes during construction too.
    environment = std::make_unique<Environment>(this);
    auto camera = std::make_unique<PerspectiveCamera>();
    viewportCamera = std::make_shared<CameraInstance>(
        std::move(camera), "Viewport Camera", Transform(vec3(0.f, 0.f, 5.f)));
    viewportCamera->scene = this;
}

Scene::~Scene() = default;

void Scene::synchronizeBeforeMutation()
{
    gpuSyncPending_.store(true, std::memory_order_relaxed);
}

void Scene::releaseGpuResources()
{
    for (Mesh& mesh : meshes)
        mesh.releaseGpu();
    for (Texture& texture : textures)
        texture.image = {};
    for (Material& material : materials)
        material.releaseGpu();
    environment->releaseGpu();
}

void Scene::notifyGeometryChanged() {
    setDirtyFlag(TLAS);
    setDirtyFlag(Accumulation);
}

// ── Internal registration ─────────────────────────────────────────────────────

SceneObjectHandle Scene::allocateObjectSlot(const uint32_t denseIndex) {
    if (freeObjectSlots.empty()) {
        objectSlots.push_back({denseIndex, 0});
        return {static_cast<uint32_t>(objectSlots.size() - 1), 0};
    }
    const uint32_t slot = freeObjectSlots.back();
    freeObjectSlots.pop_back();
    objectSlots[slot].denseIndex = denseIndex;
    return {slot, objectSlots[slot].generation};
}

void Scene::releaseObjectSlot(const SceneObjectHandle handle) {
    if (!isValid(handle))
        return;
    objectSlots[handle.index()].denseIndex = ~0u;
    // Bumping the generation is what makes a handle to the removed object
    // resolve as stale rather than aliasing whatever reuses the slot.
    ++objectSlots[handle.index()].generation;
    freeObjectSlots.push_back(handle.index());
}

bool Scene::isValid(const SceneObjectHandle handle) const {
    return handle.index() < objectSlots.size()
        && objectSlots[handle.index()].denseIndex != ~0u
        && objectSlots[handle.index()].generation == handle.generation();
}

uint32_t Scene::registerObject(std::unique_ptr<SceneObject> sceneObject) {
    sceneObject->scene = this;
    std::shared_ptr<SceneObject> sharedObject(std::move(sceneObject));

    // Adding another camera must not unexpectedly change the rendered view.
    // The first camera is selected as a useful default; later changes are explicit.
    if (auto camera = std::dynamic_pointer_cast<CameraInstance>(sharedObject);
        camera && activeCamera.expired())
        activateCamera(camera);

    if (auto light = std::dynamic_pointer_cast<LightInstance>(sharedObject))
        registerLight(*light);

    notifyGeometryChanged();
    // The slot is published only once the object is reachable through the dense
    // array, so a handle never points at a gap.
    const uint32_t denseIndex = static_cast<uint32_t>(sceneObjects.size());
    sharedObject->setHandle(allocateObjectSlot(denseIndex));
    if (auto* meshInstance = dynamic_cast<MeshInstance*>(sharedObject.get());
        meshInstance && meshInstance->hasMesh())
        addMeshInstanceSlot(*meshInstance);
    if (!sharedObject->getParent())
        addRootCandidate(*sharedObject);
    sceneObjects.push_back(std::move(sharedObject));
    notifyHierarchyChanged();
    return denseIndex;
}

// ── Public lifetime API ───────────────────────────────────────────────────────

void Scene::clear(const bool preserveViewportState) {
    synchronizeBeforeMutation();
    std::optional<Texture> environmentTexture;
    if (preserveViewportState) {
        // A scene-owned camera is about to leave. Carry its lens and world
        // transform into the persistent viewport camera before removing it.
        if (const auto camera = activeCamera.lock()) {
            viewportCamera = std::make_shared<CameraInstance>(*camera);
            viewportCamera->name = "Viewport Camera";
            viewportCamera->setWorldTransformFromMatrix(camera->getWorldTransform().getMatrix());
        }
        const int texture = environment->getTextureIndex();
        if (texture >= 0 && static_cast<size_t>(texture) < textures.size())
            environmentTexture.emplace(std::move(textures[texture]));
    }
    // Switch rendering to the persistent viewport camera before scene-owned
    // cameras are destroyed.
    activateCamera(nullptr);
    copiedObject.reset();
    sceneObjects.clear();
    ++clearEpoch_;
    meshInstanceSlots_.clear();
    changedMeshInstanceSlots_.clear();
    meshInstanceSlotListed_.clear();
    for (auto& lights : changedLights_)
        lights.clear();
    for (auto& listed : lightListed_)
        listed.clear();
    changedMeshes_.clear();
    meshListed_.clear();
    changedMaterials_.clear();
    materialListed_.clear();
    materialsToCompile_.clear();
    materialCompileListed_.clear();
    for (auto& lights : lightObjects_)
        lights.clear();
    rootCandidates_.clear();
    notifyHierarchyChanged();
    // Retire the slots rather than dropping the table, so handles that outlive
    // the clear stay detectably stale instead of aliasing a future object.
    for (uint32_t slot = 0; slot < objectSlots.size(); ++slot) {
        if (objectSlots[slot].denseIndex == ~0u)
            continue;
        objectSlots[slot].denseIndex = ~0u;
        ++objectSlots[slot].generation;
        freeObjectSlots.push_back(slot);
    }
    meshes.clear();
    materials.clear();
    textures.clear();
    meshesByPath_.clear();
    texturesByKey_.clear();
    notifyTexturesChanged();
    pointLights.clear();
    spotLights.clear();
    rectLights.clear();
    directionalLights.clear();
    materialxSourcePaths.clear();
    materialxDocuments.clear();
    notifyMaterialChanged();
    importedFileRoots_.clear();
    assignActiveObject({});
    if (preserveViewportState) {
        if (environmentTexture)
            environment->setHdriTexture(*addTexture(std::move(*environmentTexture)));
    } else {
        renderSettings = {};
        environment->clearHdriTexture();
        environment->setColor(vec3(1.0f));
        environment->setRotation(0.0f);
        environment->setVisibleExposure(0.0f);
        environment->setLightingExposure(1.0f);
        environment->setVisibleToCamera(true);
        environment->setLowerHemisphere(vec3(0.0f), 0.0f);
        environment->setLightControls(true, 1.0f);
        environment->setEquirectangularMapping();
    }
    for (const auto flag : {TLAS, Meshes, Textures, EnvironmentCdf, Lights,
                           CameraState, Accumulation, Materials})
        setDirtyFlag(flag);
}

SceneObjectHandle Scene::add(std::unique_ptr<SceneObject> sceneObject, const SceneObjectHandle parent) {
    synchronizeBeforeMutation();
    const auto parentPtr = findObjectPtr(parent);
    const uint32_t index = registerObject(std::move(sceneObject));
    if (parentPtr)
        parentPtr->addChild(sceneObjects[index]);
    const std::shared_ptr<SceneObject> object = sceneObjects[index];
    object->onAdded();
    return object->getHandle();
}

Mesh* Scene::add(Mesh mesh, const bool reuseExisting) {
    const std::string key = mesh.getPath();
    if (reuseExisting && !key.empty()) {
        if (const auto found = meshesByPath_.find(key);
            found != meshesByPath_.end())
            return found->second;
    }
    synchronizeBeforeMutation();
    meshes.push_back(std::move(mesh));
    Mesh* result = &meshes.back();
    result->setMeshIndex(static_cast<uint32_t>(meshes.size() - 1));
    if (!key.empty())
        meshesByPath_.insert_or_assign(key, result);
    meshListed_.push_back(false);
    markMeshChanged(*result);
    return result;
}

Material* Scene::add(Material material) {
    synchronizeBeforeMutation();
    materials.push_back(std::move(material));
    const auto index = static_cast<uint32_t>(materials.size() - 1);
    materials.back().sceneIndex = index;
    // Publish a record immediately so the renderer's pointer table never
    // contains a null entry for a material whose program has not compiled yet.
    materialxSourcePaths.emplace_back();
    materialxDocuments.emplace_back();
    materialListed_.push_back(false);
    materialCompileListed_.push_back(false);
    notifyMaterialChanged();
    materials.back().revision = materialRevision_;
    markMaterialChanged(index);
    markMaterialForCompile(index);
    setDirtyFlag(Accumulation);
    return &materials.back();
}

Material* Scene::addMaterial(MaterialX::DocumentPtr material, const uint32_t flags) {
    Material record{};
    nr::graphics::Material data{};
    data.flags = flags;
    record.setData(data);
    Material* result = add(std::move(record));
    materialxDocuments.back() = std::move(material);
    notifyMaterialChanged();
    result->revision = materialRevision_;
    return result;
}

void Scene::updateMaterialDocument(
    Material* material, MaterialX::DocumentPtr document)
{
    const uint32_t index = getMaterialIndex(material);
    if (index == ~0u)
        return;
    synchronizeBeforeMutation();
    materialxDocuments[index] = std::move(document);
    materialxSourcePaths[index].clear();
    // The published program was compiled from the old document.
    material->shaderProgram = {};
    material->compiled = false;
    markMaterialChanged(index);
    markMaterialForCompile(index);
    setDirtyFlag(Accumulation);
    notifyMaterialChanged();
    material->revision = materialRevision_;
}

void Scene::invalidateMaterial(Material* material)
{
    const uint32_t index = getMaterialIndex(material);
    if (index == ~0u)
        return;
    synchronizeBeforeMutation();
    // Dropping the program is what marks the material for recompilation; its
    // GPU allocations are replaced wholesale when the new one is published.
    material->shaderProgram = {};
    material->compiled = false;
    markMaterialChanged(index);
    markMaterialForCompile(index);
    setDirtyFlag(Accumulation);
    notifyMaterialChanged();
    material->revision = materialRevision_;
}

Texture* Scene::addTexture(Texture texture) {
    const std::string key = texture.getPath().empty()
        ? texture.getName() : texture.getPath();
    if (!key.empty()) {
        if (const auto found = texturesByKey_.find(key);
            found != texturesByKey_.end())
            return found->second;
    }
    synchronizeBeforeMutation();
    textures.push_back(std::move(texture));
    Texture* result = &textures.back();
    result->sceneIndex = static_cast<int>(textures.size() - 1);
    // Upload once, here. A texture that fails to load stays a valid scene
    // texture with no image; the renderer samples its white fallback for it.
    if (!key.empty())
        texturesByKey_.insert_or_assign(key, result);
    setDirtyFlag(Textures);
    notifyTexturesChanged();
    return result;
}

void Scene::reserveForImport(
    const size_t meshCount, const size_t materialCount, const size_t objectCount)
{
    synchronizeBeforeMutation();
    materialxSourcePaths.reserve(materialxSourcePaths.size() + materialCount);
    materialxDocuments.reserve(materialxDocuments.size() + materialCount);
    sceneObjects.reserve(sceneObjects.size() + objectCount);
    objectSlots.reserve(objectSlots.size() + objectCount);
}

std::vector<std::string> Scene::getTextureNames() const {
    std::vector<std::string> names;
    names.reserve(textures.size());
    for (const Texture& texture : textures)
        names.push_back(texture.getName());
    return names;
}

void Scene::setEnvironmentTexture(Texture* texture) {
    if (texture == nullptr) {
        clearEnvironmentTexture();
        return;
    }
    environment->setHdriTexture(*texture);
}

void Scene::clearEnvironmentTexture() {
    environment->clearHdriTexture();
}

bool Scene::remove(SceneObject* objToRemove) {
    const auto root = findObjectPtr(objToRemove);
    if (!root)
        return false;

    // One pass over the subtree; each object leaves in O(1), which keeps
    // removing a whole imported map linear in its size.
    std::vector<std::shared_ptr<SceneObject>> subtree{root};
    for (std::size_t i = 0; i < subtree.size(); ++i)
        for (const auto& child : subtree[i]->children)
            if (auto locked = child.lock())
                subtree.push_back(std::move(locked));
    if (SceneObject* parent = root->getParent())
        parent->removeChild(root.get());

    const CameraInstance* activeCameraBefore = getActiveCamera();
    bool removedActiveCamera = false;
    for (const auto& object : subtree) {
        if (auto* light = dynamic_cast<LightInstance*>(object.get()))
            unregisterLight(*light);
        if (auto* meshInstance = dynamic_cast<MeshInstance*>(object.get());
            meshInstance && meshInstance->slot != ~0u)
            removeMeshInstanceSlot(*meshInstance);
        removedActiveCamera |= object.get() == activeCameraBefore;
        if (activeObject == object->getHandle())
            assignActiveObject({});

        // Object order carries no meaning, so the last object fills the hole.
        const uint32_t hole = objectSlots[object->getHandle().index()].denseIndex;
        if (hole + 1 != sceneObjects.size()) {
            sceneObjects[hole] = std::move(sceneObjects.back());
            objectSlots[sceneObjects[hole]->getHandle().index()].denseIndex = hole;
        }
        sceneObjects.pop_back();
        releaseObjectSlot(object->getHandle());
        object->scene = nullptr;
    }

    if (removedActiveCamera) {
        const auto replacement = std::ranges::find_if(sceneObjects,
            [](const std::shared_ptr<SceneObject>& object) {
                return dynamic_cast<CameraInstance*>(object.get()) != nullptr;
            });
        activateCamera(replacement != sceneObjects.end()
            ? std::static_pointer_cast<CameraInstance>(*replacement) : nullptr);
    }
    notifyGeometryChanged();
    notifyHierarchyChanged();
    return true;
}

bool Scene::removeObject(const SceneObjectHandle handle) {
    synchronizeBeforeMutation();
    return remove(findObjectPtr(handle).get());
}

bool Scene::replaceObject(SceneObject* oldObject, std::unique_ptr<SceneObject> newObject) {
    synchronizeBeforeMutation();
    if (!oldObject || !newObject)
        return false;

    if (!findObjectPtr(oldObject))
        return false;
    const uint32_t index = objectSlots[oldObject->getHandle().index()].denseIndex;
    const auto it = sceneObjects.begin() + index;
    const bool wasActiveCamera = oldObject == getActiveCamera();
    const bool replacedCamera = dynamic_cast<CameraInstance*>(oldObject) != nullptr;
    SceneObject* parent = oldObject->getParent();
    const auto parentPtr = findObjectPtr(parent);
    const auto children = oldObject->getChildren();
    const bool wasCopied = copiedObject.lock().get() == oldObject;

    if (parent)
        parent->removeChild(oldObject);
    if (auto* oldLight = dynamic_cast<LightInstance*>(oldObject))
        unregisterLight(*oldLight);
    if (auto* oldMesh = dynamic_cast<MeshInstance*>(oldObject); oldMesh && oldMesh->slot != ~0u)
        removeMeshInstanceSlot(*oldMesh);
    oldObject->clearParent();
    oldObject->children.clear();
    oldObject->scene = nullptr;

    std::shared_ptr<SceneObject> newShared(std::move(newObject));
    newShared->scene = this;
    // The replacement takes over the slot, so handles held elsewhere keep
    // resolving -- that is the point of replacing rather than remove + add.
    newShared->setHandle(oldObject->getHandle());

    if (auto* newLight = dynamic_cast<LightInstance*>(newShared.get()))
        registerLight(*newLight);
    if (auto* newMesh = dynamic_cast<MeshInstance*>(newShared.get()); newMesh && newMesh->hasMesh())
        addMeshInstanceSlot(*newMesh);

    if (wasActiveCamera || activeCamera.expired())
        activateCamera(std::dynamic_pointer_cast<CameraInstance>(newShared));

    *it = std::move(newShared);

    if (replacedCamera || dynamic_cast<CameraInstance*>(sceneObjects[index].get()))
        setDirtyFlag(CameraState);

    if (parentPtr)
        parentPtr->addChild(sceneObjects[index]);
    else
        addRootCandidate(*sceneObjects[index]);
    for (const auto& child : children)
        sceneObjects[index]->addChild(child);
    if (wasCopied)
        copiedObject = sceneObjects[index];

    notifyGeometryChanged();
    notifyHierarchyChanged();
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
    for (SceneListener* listener : listeners_)
        listener->onActiveCameraChanged();
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

bool Scene::setActiveObject(const SceneObjectHandle handle) {
    if (handle.isValid() && !isValid(handle))
        return false;
    assignActiveObject(handle.isValid() ? handle : SceneObjectHandle{});
    return true;
}

void Scene::assignActiveObject(const SceneObjectHandle handle) {
    if (activeObject == handle)
        return;
    activeObject = handle;
    for (SceneListener* listener : listeners_)
        listener->onActiveObjectChanged();
}

void Scene::notifyHierarchyChanged() {
    ++hierarchyRevision;
    for (SceneListener* listener : listeners_)
        listener->onHierarchyChanged();
}

void Scene::notifyTexturesChanged() {
    ++textureRevision_;
    for (SceneListener* listener : listeners_)
        listener->onTexturesChanged();
}

void Scene::notifyObjectTransformChanged(const SceneObject& object) {
    for (SceneListener* listener : listeners_)
        listener->onObjectTransformChanged(object.getHandle());
}

// ── Light management ─────────────────────────────────────────────────────────

uint32_t Scene::registerLight(LightInstance& light)
{
    uint32_t idx = UINT32_MAX;
    switch (light.getLightType()) {
    case LightInstance::TypePoint:
        idx = static_cast<uint32_t>(pointLights.size());
        pointLights.push_back(static_cast<PointLightInstance&>(light).getData());
        break;
    case LightInstance::TypeSpot:
        idx = static_cast<uint32_t>(spotLights.size());
        spotLights.push_back(static_cast<SpotLightInstance&>(light).getData());
        break;
    case LightInstance::TypeRect:
        idx = static_cast<uint32_t>(rectLights.size());
        rectLights.push_back(static_cast<RectLightInstance&>(light).getData());
        break;
    case LightInstance::TypeDirectional:
        idx = static_cast<uint32_t>(directionalLights.size());
        directionalLights.push_back(static_cast<DirectionalLightInstance&>(light).getData());
        break;
    }
    light.setLightIndex(idx);
    // The pushed record holds authored lengths; this scales them to the world.
    light.commitLightChanges();
    lightObjects_[light.getLightType()].push_back(&light);
    setDirtyFlag(Lights);
    setDirtyFlag(Accumulation);
    return idx;
}

void Scene::unregisterLight(LightInstance& light)
{
    const uint32_t idx = light.getLightIndex();
    const auto removeRecord = [idx](auto& records) {
        if (idx + 1 != records.size())
            records[idx] = std::move(records.back());
        records.pop_back();
    };
    switch (light.getLightType()) {
    case LightInstance::TypePoint: removeRecord(pointLights); break;
    case LightInstance::TypeSpot: removeRecord(spotLights); break;
    case LightInstance::TypeRect: removeRecord(rectLights); break;
    case LightInstance::TypeDirectional: removeRecord(directionalLights); break;
    }
    // The objects mirror the records, so the displaced record's object is
    // the one that moves along with it.
    auto& objects = lightObjects_[light.getLightType()];
    if (idx + 1 != objects.size()) {
        objects[idx] = objects.back();
        objects[idx]->setLightIndex(idx);
    }
    objects.pop_back();
    light.setLightIndex(UINT32_MAX);
    setDirtyFlag(Lights);
    setDirtyFlag(Accumulation);
}

const LightInstance& Scene::getLightObject(const int lightType, const uint32_t lightIndex) const
{
    return *lightObjects_[lightType][lightIndex];
}

std::vector<const LightInstance*> Scene::getLightObjects() const
{
    std::vector<const LightInstance*> result;
    for (const auto& objects : lightObjects_)
        result.insert(result.end(), objects.begin(), objects.end());
    return result;
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
        addRootCandidate(*objectToMove);
        objectToMove->setLocalTransform(Transform{oldWorldMatrix});
    }
    notifyHierarchyChanged();
}

bool Scene::reparentObject(
    const SceneObjectHandle handle, const SceneObjectHandle newParentHandle) {
    synchronizeBeforeMutation();
    const auto objectToMove = findObjectPtr(handle);
    const auto newParent = findObjectPtr(newParentHandle);
    if (!objectToMove || (newParentHandle.isValid() && !newParent))
        return false;

    reparent(objectToMove.get(), newParent.get());
    return true;
}

// ── Clipboard ────────────────────────────────────────────────────────────────

void Scene::copyObject(const SceneObjectHandle handle) {
    copiedObject = findObjectPtr(handle);
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
    setActiveObject(newObject->getHandle());
}

// ── Queries ───────────────────────────────────────────────────────────────────

std::vector<std::shared_ptr<SceneObject>> Scene::getRootObjects() const {
    std::vector<std::shared_ptr<SceneObject>> result;
    std::erase_if(rootCandidates_, [this, &result](const SceneObjectHandle handle) {
        auto object = findObjectPtr(handle);
        if (!object)
            return true;
        if (object->getParent()) {
            object->rootCandidate = false;
            return true;
        }
        result.push_back(std::move(object));
        return false;
    });
    return result;
}

void Scene::addRootCandidate(SceneObject& object) const {
    if (std::exchange(object.rootCandidate, true))
        return;
    rootCandidates_.push_back(object.getHandle());
}

void Scene::addMeshInstanceSlot(MeshInstance& instance) {
    instance.slot = static_cast<uint32_t>(meshInstanceSlots_.size());
    meshInstanceSlots_.push_back(&instance);
    meshInstanceSlotListed_.push_back(false);
    markMeshInstanceChanged(instance.slot);
}

void Scene::removeMeshInstanceSlot(MeshInstance& instance) {
    const uint32_t hole = instance.slot;
    const uint32_t last = static_cast<uint32_t>(meshInstanceSlots_.size() - 1);
    if (hole != last) {
        meshInstanceSlots_[hole] = meshInstanceSlots_[last];
        meshInstanceSlots_[hole]->slot = hole;
        markMeshInstanceChanged(hole);
    }
    meshInstanceSlots_.pop_back();
    meshInstanceSlotListed_.pop_back();
    instance.slot = ~0u;
}

void Scene::markMeshInstanceChanged(const uint32_t slot) {
    if (meshInstanceSlotListed_[slot])
        return;
    meshInstanceSlotListed_[slot] = true;
    changedMeshInstanceSlots_.push_back(slot);
}

void Scene::markMeshChanged(Mesh& mesh) {
    // A mesh outside the scene is listed when add() takes it.
    if (mesh.getMeshIndex() >= meshListed_.size() || meshListed_[mesh.getMeshIndex()])
        return;
    meshListed_[mesh.getMeshIndex()] = true;
    changedMeshes_.push_back(&mesh);
    setDirtyFlag(Meshes);
}

void Scene::markMaterialChanged(const uint32_t materialIndex) {
    setDirtyFlag(Materials);
    if (materialListed_[materialIndex])
        return;
    materialListed_[materialIndex] = true;
    changedMaterials_.push_back(materialIndex);
}

void Scene::markMaterialForCompile(const uint32_t materialIndex) {
    if (materialCompileListed_[materialIndex])
        return;
    materialCompileListed_[materialIndex] = true;
    materialsToCompile_.push_back(materialIndex);
}

std::vector<uint32_t> Scene::takeChangedMeshInstanceSlots() {
    // A slot listed before a removal shrank the table no longer exists.
    std::erase_if(changedMeshInstanceSlots_, [this](const uint32_t slot) {
        return slot >= meshInstanceSlots_.size();
    });
    for (const uint32_t slot : changedMeshInstanceSlots_)
        meshInstanceSlotListed_[slot] = false;
    return std::exchange(changedMeshInstanceSlots_, {});
}

void Scene::markLightChanged(const int lightType, const uint32_t lightIndex) {
    std::vector<bool>& listed = lightListed_[lightType];
    if (lightIndex >= listed.size())
        listed.resize(lightIndex + 1);
    if (listed[lightIndex])
        return;
    listed[lightIndex] = true;
    changedLights_[lightType].push_back(lightIndex);
}

Scene::LightIndices Scene::takeChangedLights() {
    LightIndices changed = std::exchange(changedLights_, {});
    for (size_t type = 0; type < changed.size(); ++type) {
        for (const uint32_t index : changed[type])
            lightListed_[type][index] = false;
        // A record listed before a removal shrank the table no longer exists.
        std::erase_if(changed[type], [this, type](const uint32_t index) {
            return index >= lightObjects_[type].size();
        });
    }
    return changed;
}

std::vector<Mesh*> Scene::takeChangedMeshes() {
    for (const Mesh* mesh : changedMeshes_)
        meshListed_[mesh->getMeshIndex()] = false;
    return std::exchange(changedMeshes_, {});
}

std::vector<uint32_t> Scene::takeChangedMaterials() {
    for (const uint32_t index : changedMaterials_)
        materialListed_[index] = false;
    return std::exchange(changedMaterials_, {});
}

std::vector<uint32_t> Scene::takeMaterialsToCompile() {
    for (const uint32_t index : materialsToCompile_)
        materialCompileListed_[index] = false;
    return std::exchange(materialsToCompile_, {});
}

uint32_t Scene::getActiveCryptomatteId() const
{
    // Mesh instances are numbered by their slot, as in the TLAS.
    const auto* mesh = dynamic_cast<const MeshInstance*>(getObject(activeObject));
    return mesh ? mesh->slot : ~0u;
}

SceneObject* Scene::findCryptomatteObject(const uint32_t id) const
{
    return id < meshInstanceSlots_.size() ? meshInstanceSlots_[id] : nullptr;
}

Texture* Scene::findTexture(const std::string& key) const {
    const auto found = texturesByKey_.find(key);
    return found != texturesByKey_.end() ? found->second : nullptr;
}

Mesh* Scene::findMesh(const std::string& path) const {
    const auto found = meshesByPath_.find(path);
    return found != meshesByPath_.end() ? found->second : nullptr;
}

uint32_t Scene::getMaterialIndex(const Material* material) const
{
    if (material == nullptr || material->sceneIndex >= materials.size()
        || &materials[material->sceneIndex] != material)
        return ~0u;
    return material->sceneIndex;
}

SceneObjectHandle Scene::findImportedFileRoot(const std::string& resolvedPath) const {
    const auto found = importedFileRoots_.find(resolvedPath);
    if (found == importedFileRoots_.end() || !isValid(found->second))
        return {};
    return found->second;
}

void Scene::registerImportedFileRoot(
    const std::string& resolvedPath, const SceneObjectHandle handle)
{
    importedFileRoots_[resolvedPath] = handle;
}

// ── Lookups ───────────────────────────────────────────────────────────────────

std::shared_ptr<SceneObject> Scene::findObjectPtr(const SceneObject* object) const {
    if (!object || object->scene != this)
        return nullptr;
    auto found = findObjectPtr(object->getHandle());
    return found.get() == object ? found : nullptr;
}

std::shared_ptr<SceneObject> Scene::findObjectPtr(const SceneObjectHandle handle) const {
    if (!isValid(handle))
        return nullptr;
    return sceneObjects[objectSlots[handle.index()].denseIndex];
}

void Scene::setMaterialProgram(const std::size_t materialIndex,
    MaterialShaderProgram shaderProgram)
{
    synchronizeBeforeMutation();
    Material& material = materials[materialIndex];
    material.releaseGpu();
    material.shaderProgram = std::move(shaderProgram);
    material.compiled = true;
    markMaterialChanged(static_cast<uint32_t>(materialIndex));
    setDirtyFlag(Accumulation);
}
