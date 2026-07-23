#include "NoorRaySession.h"
#include "Scene/SceneObject.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("scene replacement preserves hierarchy and detached objects remain usable", "[scene][lifetime]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;

    const uint64_t parentId = scene.add(std::make_unique<SceneObject>(
        "Parent", Transform(glm::vec3(1.0f, 0.0f, 0.0f))));
    const uint64_t childId = scene.add(std::make_unique<SceneObject>(
        "Child", Transform(glm::vec3(0.0f, 2.0f, 0.0f))));
    REQUIRE(scene.reparentObject(childId, parentId));

    const std::shared_ptr<SceneObject> original = scene.getObjectPtr(parentId);
    REQUIRE(scene.replaceObject(original.get(), std::make_unique<SceneObject>(
        "Replacement", Transform(glm::vec3(1.0f, 0.0f, 0.0f)))));

    const std::shared_ptr<SceneObject> replacement = scene.getObjectPtr(parentId);
    REQUIRE(replacement);
    REQUIRE(replacement->getChildren().size() == 1);
    CHECK(replacement->getChildren().front()->getId() == childId);
    CHECK(original->getParent() == nullptr);

    original->setPosition(glm::vec3(3.0f));
    CHECK(original->getPosition() == glm::vec3(3.0f));
}
