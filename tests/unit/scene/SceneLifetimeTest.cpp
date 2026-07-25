#include "NoorRaySession.h"
#include "Mesh/Assets/GaussianAsset.h"
#include "Mesh/Assets/MeshAsset.h"
#include "Raytracing/Runtime/Raytracer.h"
#include "Scene/GaussianInstance.h"
#include "Scene/MeshInstance.h"
#include "Scene/SceneObject.h"

#include <cuda_runtime_api.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("scene replacement preserves hierarchy and detached objects remain usable", "[scene][lifetime]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    const SceneObjectHandle parent = scene.add(std::make_unique<SceneObject>(
        "Parent", Transform(glm::vec3(1.0f, 0.0f, 0.0f))));
    const SceneObjectHandle child = scene.add(std::make_unique<SceneObject>(
        "Child", Transform(glm::vec3(0.0f, 2.0f, 0.0f))));
    REQUIRE(scene.reparentObject(child, parent));

    const std::shared_ptr<SceneObject> original = scene.getObjectPtr(parent);
    REQUIRE(scene.replaceObject(original.get(), std::make_unique<SceneObject>(
        "Replacement", Transform(glm::vec3(1.0f, 0.0f, 0.0f)))));

    const std::shared_ptr<SceneObject> replacement = scene.getObjectPtr(parent);
    REQUIRE(replacement);
    REQUIRE(replacement->getChildren().size() == 1);
    CHECK(replacement->getChildren().front()->getHandle() == child);
    CHECK(original->getParent() == nullptr);

    original->setPosition(glm::vec3(3.0f));
    CHECK(original->getPosition() == glm::vec3(3.0f));
}

TEST_CASE("object handles survive unrelated removals and go stale on their own", "[scene][handle]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    const SceneObjectHandle first = scene.add(
        std::make_unique<SceneObject>("First", Transform{}));
    const SceneObjectHandle second = scene.add(
        std::make_unique<SceneObject>("Second", Transform{}));
    const SceneObjectHandle third = scene.add(
        std::make_unique<SceneObject>("Third", Transform{}));

    // Removing from the middle shifts the dense storage but must not disturb
    // the handles of the objects around it.
    REQUIRE(scene.removeObject(second));
    CHECK_FALSE(scene.isValid(second));
    REQUIRE(scene.getObject(first) != nullptr);
    REQUIRE(scene.getObject(third) != nullptr);
    CHECK(scene.getObject(first)->getName() == "First");
    CHECK(scene.getObject(third)->getName() == "Third");

    // The freed slot is recycled, and the stale handle must not follow it.
    const SceneObjectHandle recycled = scene.add(
        std::make_unique<SceneObject>("Fourth", Transform{}));
    CHECK(recycled.index() == second.index());
    CHECK(recycled != second);
    CHECK_FALSE(scene.isValid(second));
    CHECK(scene.getObject(second) == nullptr);
    CHECK(scene.getObject(recycled)->getName() == "Fourth");
}

TEST_CASE("assets are reclaimed once the last instance referencing them is gone", "[scene][handle]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    std::vector<Gaussian> gaussians(16);
    GaussianAssetRef asset = scene.add(
        GaussianAsset(scene, "Splats", std::move(gaussians)));
    const GaussianAssetHandle assetHandle = asset.handle();

    const SceneObjectHandle instance = scene.add(
        std::make_unique<GaussianInstance>(scene, "Splat", asset, Transform{}));
    const SceneObjectHandle clone = scene.add(
        std::make_unique<GaussianInstance>(scene, "Splat Copy", asset, Transform{}));

    // The importer's reference plus one per instance.
    REQUIRE(scene.getGaussianAssetRegistry().useCount(assetHandle) == 3);

    REQUIRE(scene.removeObject(instance));
    CHECK(scene.getGaussianAsset(assetHandle) != nullptr);
    CHECK(scene.getGaussianAsset(assetHandle)->getGaussianCount() == 16);

    REQUIRE(scene.removeObject(clone));
    CHECK(scene.getGaussianAssetRegistry().useCount(assetHandle) == 1);

    // Dropping the last reference frees the splat data and retires the slot.
    asset.reset();
    CHECK(scene.getGaussianAsset(assetHandle) == nullptr);
}

TEST_CASE("removing the last splat instance releases its device memory", "[scene][gaussian]")
{
    // Splat data and the per-instance acceleration structure built from it are
    // the largest device allocations a scene makes, and deleting the instance
    // used to leave both resident.
    const auto deviceBytesInUse = [] {
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        REQUIRE(cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess);
        return totalBytes - freeBytes;
    };

    const auto makeGaussians = [](const uint32_t count) {
        std::vector<Gaussian> gaussians(count);
        for (uint32_t index = 0; index < count; ++index) {
            Gaussian& gaussian = gaussians[index];
            gaussian.transform = glm::mat4x3(
                glm::vec3(0.01f, 0.0f, 0.0f), glm::vec3(0.0f, 0.01f, 0.0f),
                glm::vec3(0.0f, 0.0f, 0.01f),
                glm::vec3(static_cast<float>(index % 1000) * 0.05f,
                    static_cast<float>((index / 1000) % 1000) * 0.05f, 0.0f));
            gaussian.opacity = 0.5f;
            gaussian.sphericalHarmonics.count = 1;
            gaussian.setShCoefficient(0, glm::vec3(0.5f));
        }
        return gaussians;
    };

    noorray::NoorRaySession session;
    Scene& scene = session.scene;
    session.raytracer->resize(64, 64);

    const auto renderOnce = [&] {
        session.raytracer->renderFrame();
        session.raytracer->waitForRender();
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    };

    // Loads the splat, renders it, removes it again and reports device usage
    // while it was resident.
    const auto loadRenderRemove = [&](const uint32_t count) {
        SceneObjectHandle instance;
        GaussianAssetHandle assetHandle;
        {
            const GaussianAssetRef asset = scene.add(
                GaussianAsset(scene, "Splats", makeGaussians(count)));
            assetHandle = asset.handle();
            instance = scene.add(std::make_unique<GaussianInstance>(
                scene, "Splat", asset, Transform{}));
        }
        renderOnce();
        const size_t loaded = deviceBytesInUse();

        REQUIRE(scene.removeObject(instance));
        renderOnce();

        // Nothing refers to the asset any more, so its slot and its splat data
        // are gone and the flattened GPU arrays are deallocated rather than
        // merely emptied.
        CHECK(scene.getGaussianAsset(assetHandle) == nullptr);
        CHECK(scene.getGaussianCount() == 0);
        CHECK(scene.getGaussianShCoeffs() == nullptr);
        CHECK(scene.getGaussianOpacities() == nullptr);
        return loaded;
    };

    // A throwaway cycle with a negligible splat pays the one-time cost of the
    // Gaussian pipeline and its shader binding table, so the idle level
    // measured afterwards is a clean baseline for the real one.
    renderOnce();
    loadRenderRemove(1'000);
    const size_t idleBefore = deviceBytesInUse();

    const size_t loaded = loadRenderRemove(2'000'000);
    const size_t idleAfter = deviceBytesInUse();
    REQUIRE(loaded > idleBefore);

    const size_t splatBytes = loaded - idleBefore;
    const size_t retained = idleAfter > idleBefore ? idleAfter - idleBefore : 0;
    CHECK(retained * 4 < splatBytes);
}
