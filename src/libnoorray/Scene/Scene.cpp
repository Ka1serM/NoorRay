#include "Scene.h"
#include <algorithm>
#include "Rendering/Camera/CameraInstance.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Rendering/Lighting/PointLight.h"
#include "Rendering/Lighting/RectLight.h"
#include "Rendering/Lighting/SpotLight.h"
#include "Rendering/Lighting/DirectionalLight.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "Scene/SceneObject.h"
#include "Scene/Import/SceneImporter.h"
#include "Scene/Import/SceneReader.h"

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
    gpuSyncPending_.store(true, std::memory_order_relaxed);
    reclaimUnusedResources();
}

void Scene::reclaimUnusedResources()
{
    // A mesh asset releasing its material references reclaims those materials
    // inside the registry, without the Scene being told. Dropping the matching
    // texture references here is what lets the texture memory go too. Consume
    // only release events instead of walking the complete material high-water
    // mark before every Scene mutation.
    materials.consumeReleasedSlots([this](const uint32_t slot) {
        if (slot < materialxSourcePaths.size())
            materialxSourcePaths[slot].clear();
        if (slot < materialxDocuments.size())
            materialxDocuments[slot].reset();
    });
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
    const auto gaussianInstance = std::dynamic_pointer_cast<GaussianInstance>(sharedObject);
    if (gaussianInstance)
        setDirtyFlag(GaussianData);

    notifyGeometryChanged();
    // The slot is published only once the object is reachable through the dense
    // array, so a handle never points at a gap.
    const uint32_t denseIndex = static_cast<uint32_t>(sceneObjects.size());
    sharedObject->setHandle(allocateObjectSlot(denseIndex));
    sceneObjects.push_back(std::move(sharedObject));
    if (gaussianInstance && gaussianInstance->hasGaussianAsset())
    {
        gaussianInstance->sceneInstanceIndex = static_cast<uint32_t>(gaussianInstances.size());
        gaussianCount += gaussianInstance->getGaussianAsset().getGaussianCount();
        gaussianInstances.push_back(gaussianInstance);
        dirtyGaussianInstanceFlags.push_back(0);
    }
    return denseIndex;
}

void Scene::rebuildGaussianInstanceCache()
{
    gaussianInstances.clear();
    gaussianCount = 0;
    for (const auto& object : sceneObjects)
    {
        if (auto instance = std::dynamic_pointer_cast<GaussianInstance>(object))
        {
            // An instance whose asset was reclaimed has nothing to render, so
            // it stays out of the flattened splat data and the TLAS.
            if (!instance->hasGaussianAsset())
                continue;
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
    copiedObject.reset();
    // Objects own asset references, so they go first: by the time the
    // registries are cleared almost everything has already been reclaimed.
    sceneObjects.clear();
    // Retire the slots rather than dropping the table, so handles that outlive
    // the clear stay detectably stale instead of aliasing a future object.
    for (uint32_t slot = 0; slot < objectSlots.size(); ++slot) {
        if (objectSlots[slot].denseIndex == ~0u)
            continue;
        objectSlots[slot].denseIndex = ~0u;
        ++objectSlots[slot].generation;
        freeObjectSlots.push_back(slot);
    }
    gaussianInstances.clear();
    gaussianCount = 0;
    environmentTexture.reset();
    materialOwners.clear();
    meshAssets.clear();
    materials.clear();
    gaussianAssets.clear();
    textures.clear();
    pointLights.clear();
    spotLights.clear();
    rectLights.clear();
    directionalLights.clear();
    dirtyMeshInstanceIndices.clear();
    dirtyGaussianInstanceIndices.clear();
    dirtyGaussianInstanceFlags.clear();
    gaussianOpacities.clear();
    gaussianShCoeffs.clear();
    gaussianInstanceOffsets.clear();
    materialxSourcePaths.clear();
    materialxDocuments.clear();
    importedFileRoots_.clear();
    activeObject = {};
    renderSettings = {};
    environment->destroyCdf();
    environment->textureIndex = -1;
    environment->color = vec3(1.0f);
    environment->rotation = 0.0f;
    environment->visibleExposure = 0.0f;
    environment->lightingExposure = 1.0f;
    environment->setEquirectangularMapping();
    environment->updateDerivedSettings();
    dirtyFlags = TLAS | Meshes | Textures | EnvironmentCdf | Lights
        | CameraState | Accumulation | GaussianData;
    ++lightRevision;
}

SceneObjectHandle Scene::add(std::unique_ptr<SceneObject> sceneObject) {
    synchronizeBeforeMutation();
    const uint32_t index = registerObject(std::move(sceneObject));
    return sceneObjects[index]->getHandle();
}

MeshAssetRef Scene::add(MeshAsset meshAsset) {
    synchronizeBeforeMutation();
    const MeshAssetHandle handle = meshAssets.emplace(std::move(meshAsset));
    // The asset caches its own slot index; the shading kernels read it back
    // out of the mesh when they resolve a hit.
    meshAssets[handle].setMeshIndex(handle.index());
    setDirtyFlag(Meshes);
    return {meshAssets, handle};
}

MaterialRef Scene::add(Material material) {
    synchronizeBeforeMutation();
    const MaterialHandle handle = materials.emplace(material);
    materialxSourcePaths.emplace_back();
    materialxDocuments.emplace_back();
    setDirtyFlag(Meshes);
    setDirtyFlag(Accumulation);
    return {materials, handle};
}

MaterialRef Scene::addMaterial(MaterialX::DocumentPtr material) {
    MaterialRef ref = add(Material{});
    materialOwners.push_back(ref);
    materialxDocuments[ref.index()] = std::move(material);
    return ref;
}

void Scene::updateMaterialDocument(
    const MaterialHandle handle, MaterialX::DocumentPtr document)
{
    if (!materials.isValid(handle))
        return;
    synchronizeBeforeMutation();
    if (handle.index() >= materialxDocuments.size())
        materialxDocuments.resize(handle.index() + 1);
    materialxDocuments[handle.index()] = std::move(document);
    if (handle.index() < materialxSourcePaths.size())
        materialxSourcePaths[handle.index()].clear();
    setDirtyFlag(Meshes);
    setDirtyFlag(Accumulation);
}

void Scene::updateMaterial(
    const MaterialHandle handle, const Material& material)
{
    if (!materials.isValid(handle))
        return;
    synchronizeBeforeMutation();
    materials[handle] = material;
    // GpuSceneData::materials points straight at this same storage (see
    // Raytracer::updateMeshes), but Meshes must still be marked dirty so the
    // pointer/count are refreshed after materials.emplace() might have
    // reallocated, even though mesh topology itself is unchanged.
    setDirtyFlag(Meshes);
    setDirtyFlag(Accumulation);
}

void Scene::invalidateMaterial(const MaterialHandle handle)
{
    if (!materials.isValid(handle))
        return;
    synchronizeBeforeMutation();
    materials[handle].svmBytecodeOffset = 0;
    materials[handle].svmBytecodeLength = 0;
    materials[handle].svmTextureOffset = 0;
    materials[handle].svmTextureCount = 0;
    setDirtyFlag(Meshes);
    setDirtyFlag(Accumulation);
}

GaussianAssetRef Scene::add(GaussianAsset gaussianAsset) {
    synchronizeBeforeMutation();
    const GaussianAssetHandle handle =
        gaussianAssets.emplace(std::move(gaussianAsset));
    setDirtyFlag(TLAS);
    setDirtyFlag(GaussianData);
    return {gaussianAssets, handle};
}

TextureRef Scene::add(Texture texture) {
    synchronizeBeforeMutation();
    const TextureHandle handle = textures.emplace(std::move(texture));
    textures[handle].sceneIndex = static_cast<int>(handle.index());
    setDirtyFlag(Textures);
    return {textures, handle};
}

void Scene::reserveForImport(
    const size_t meshCount, const size_t materialCount, const size_t objectCount)
{
    synchronizeBeforeMutation();
    meshAssets.reserveAdditional(meshCount);
    materials.reserveAdditional(materialCount);
    materialxSourcePaths.reserve(materialxSourcePaths.size() + materialCount);
    materialxDocuments.reserve(materialxDocuments.size() + materialCount);
    sceneObjects.reserve(sceneObjects.size() + objectCount);
    objectSlots.reserve(objectSlots.size() + objectCount);
}

std::vector<std::string> Scene::getTextureNames() const {
    std::vector<std::string> names;
    names.reserve(textures.storage().size());
    for (const Texture& texture : textures.storage())
        names.push_back(texture.getName());
    return names;
}

void Scene::setEnvironmentTexture(const TextureRef& texture) {
    const Texture* resolved = texture.get();
    if (resolved == nullptr) {
        clearEnvironmentTexture();
        return;
    }
    environmentTexture = texture;
    environment->setHdriTexture(*resolved);
    setDirtyFlag(EnvironmentCdf);
    setDirtyFlag(Accumulation);
}

void Scene::clearEnvironmentTexture() {
    environmentTexture.reset();
    environment->clearHdriTexture();
    setDirtyFlag(EnvironmentCdf);
    setDirtyFlag(Accumulation);
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

    if (activeObject == (*it)->getHandle())
        activeObject = {};
    if (objToRemove == getActiveCamera()) {
        const auto replacement = std::ranges::find_if(sceneObjects,
            [objToRemove](const std::shared_ptr<SceneObject>& object) {
                return object.get() != objToRemove
                    && dynamic_cast<CameraInstance*>(object.get()) != nullptr;
            });
        activateCamera(replacement != sceneObjects.end()
            ? std::static_pointer_cast<CameraInstance>(*replacement) : nullptr);
    }

    releaseObjectSlot(objToRemove->getHandle());
    objToRemove->scene = nullptr;
    const auto erased = sceneObjects.erase(it);
    // Erasing shifts everything behind the hole down by one; the slot table is
    // the only thing that knows where an object lives, so it follows along.
    for (auto follower = erased; follower != sceneObjects.end(); ++follower)
        objectSlots[(*follower)->getHandle().index()].denseIndex =
            static_cast<uint32_t>(std::distance(sceneObjects.begin(), follower));
    rebuildGaussianInstanceCache();
    notifyGeometryChanged();
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
    // The replacement takes over the slot, so handles held elsewhere keep
    // resolving -- that is the point of replacing rather than remove + add.
    newShared->setHandle(oldObject->getHandle());

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
    for (const auto& obj : sceneObjects)
        if (obj->getParent() == nullptr)
            result.push_back(obj);
    return result;
}

std::vector<std::shared_ptr<MeshInstance>> Scene::getMeshInstances() const {
    std::vector<std::shared_ptr<MeshInstance>> result;
    for (const auto& obj : sceneObjects)
        if (auto mi = std::dynamic_pointer_cast<MeshInstance>(obj); mi && mi->hasMeshAsset())
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
        // Must skip exactly what getMeshInstances() skips: this index addresses
        // the list that builds the TLAS.
        const auto mesh = std::dynamic_pointer_cast<MeshInstance>(obj);
        if (!mesh || !mesh->hasMeshAsset())
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
    const auto& gaussianInstances = getGaussianInstances();
    gaussianInstanceOffsets.resize(gaussianInstances.size());
    uint32_t total = 0;
    for (size_t instanceIndex = 0; instanceIndex < gaussianInstances.size(); ++instanceIndex)
    {
        const auto& instance = gaussianInstances[instanceIndex];
        const GaussianAsset& asset = instance->getGaussianAsset();
        gaussianInstanceOffsets[instanceIndex] = total;
        const auto& gaussians = asset.getGaussians();
        spans.push_back({gaussians.data(), total});
        total += static_cast<uint32_t>(gaussians.size());
    }

    const uint32_t coefficientsPerGaussian = sphericalHarmonicsCoefficientCount(
        renderSettings.gaussianRenderSphericalHarmonics);
    gaussianShCoefficientCount = coefficientsPerGaussian;
    if (total == 0)
    {
        // The packed SH array is the largest managed allocation in a splat
        // scene. resize(0) would keep its capacity, so hand the memory back.
        gaussianOpacities = nr::rstd::vector<float>{};
        gaussianShCoeffs = nr::rstd::vector<__half>{};
        gaussianInstanceOffsets = nr::rstd::vector<uint32_t>{};
        return;
    }
    gaussianOpacities.resize(total);
    gaussianShCoeffs.resize(static_cast<size_t>(total)
        * coefficientsPerGaussian * SphericalHarmonicsChannelCount);

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
            __half* coefficients = gaussianShCoeffs.data()
                + static_cast<size_t>(globalIndex) * coefficientsPerGaussian
                    * SphericalHarmonicsChannelCount;
            std::fill_n(coefficients,
                coefficientsPerGaussian * SphericalHarmonicsChannelCount, __half{});
            const uint32_t count = std::min(
                gaussian.sphericalHarmonics.count, coefficientsPerGaussian);
            const __half* source = gaussian.sphericalHarmonics.values.data();
            for (uint32_t coefficient = 0; coefficient < count; ++coefficient)
            {
                std::copy_n(source + coefficient * 3, 3,
                    coefficients + coefficient * 3);
            }
        }
    });
}

uint32_t Scene::getActiveCryptomatteId(const uint32_t selectedGaussianIndex) const
{
    // Cryptomatte ids follow the same ordering as the TLAS, so instances whose
    // asset has been reclaimed are skipped here exactly as they are there.
    uint32_t meshInstanceCount = 0;
    for (const auto& object : sceneObjects)
        if (const auto mesh = std::dynamic_pointer_cast<MeshInstance>(object);
            mesh && mesh->hasMeshAsset())
            ++meshInstanceCount;

    uint32_t meshIndex = 0;
    uint32_t gaussianOffset = 0;
    for (const auto& object : sceneObjects)
    {
        if (auto mesh = std::dynamic_pointer_cast<MeshInstance>(object))
        {
            if (!mesh->hasMeshAsset())
                continue;
            if (object->getHandle() == activeObject)
                return meshIndex;
            ++meshIndex;
        }
        else if (auto gaussian = std::dynamic_pointer_cast<GaussianInstance>(object))
        {
            if (!gaussian->hasGaussianAsset())
                continue;
            if (object->getHandle() == activeObject)
            {
                const uint32_t gaussianCount = gaussian->getGaussianAsset().getGaussianCount();
                // The picker and transform gizmo use flattened Gaussian
                // indices; Cryptomatte places those after all mesh instances.
                if (selectedGaussianIndex >= gaussianOffset
                    && selectedGaussianIndex < gaussianOffset + gaussianCount)
                {
                    return meshInstanceCount + selectedGaussianIndex;
                }
                return meshInstanceCount + gaussianOffset;
            }
            gaussianOffset += gaussian->getGaussianAsset().getGaussianCount();
        }
    }
    return ~0u;
}

MeshAssetHandle Scene::findMeshAsset(const std::string& path) const {
    for (uint32_t slot = 0; slot < meshAssets.slotCount(); ++slot)
        if (meshAssets.isLiveSlot(slot) && meshAssets.storage()[slot].getPath() == path)
            return meshAssets.handleAt(slot);
    return {};
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
    if (!object) return nullptr;
    const auto it = std::ranges::find_if(sceneObjects, [object](const auto& ptr) {
        return ptr.get() == object;
    });
    return it != sceneObjects.end() ? *it : nullptr;
}

std::shared_ptr<SceneObject> Scene::findObjectPtr(const SceneObjectHandle handle) const {
    if (!isValid(handle))
        return nullptr;
    return sceneObjects[objectSlots[handle.index()].denseIndex];
}
