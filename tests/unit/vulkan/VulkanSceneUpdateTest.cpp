#include "Backend/Vulkan/Raytracer/VulkanScene.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Scene/Scene.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Vulkan scene refits fixed-topology editor transforms", "[gpu][vulkan]")
{
    gpu::Device device({.enable_validation = true,
        .application_name = "Vulkan scene update test"});
    if (!device.features().ray_tracing)
        SKIP("selected Vulkan device does not expose ray tracing");

    Scene scene;
    const auto material = nr::materialx::documentFromAuthoring(MaterialAuthoring{});
    const auto asset = scene.add(MeshAsset::CreateSphere(scene, "Sphere", material, 4, 8));
    auto object = std::make_unique<MeshInstance>(scene, "Sphere", asset, Transform{});
    MeshInstance* instance = object.get();
    scene.add(std::move(object));

    VulkanScene nativeScene(device, scene);
    const gpu::AccelerationStructureHandle original = nativeScene.topLevelHandle();
    REQUIRE(original);

    instance->setPosition({2.0f, -1.0f, 0.5f});
    REQUIRE(nativeScene.updateMutableData(scene));
    CHECK(nativeScene.topLevelHandle().value == original.value);

    for (uint32_t edit = 0; edit < 256; ++edit)
    {
        instance->setPosition({static_cast<float>(edit % 17u) * 0.01f,
            static_cast<float>(edit % 11u) * -0.01f, 0.5f});
        REQUIRE(nativeScene.updateMutableData(scene));
        REQUIRE(nativeScene.topLevelHandle().value == original.value);
    }

    scene.add(std::make_unique<MeshInstance>(scene, "Second sphere", asset, Transform{}));
    CHECK_FALSE(nativeScene.updateMutableData(scene));
}

TEST_CASE("Vulkan scene updates Gaussian values and transforms in place", "[gpu][vulkan]")
{
    gpu::Device device({.enable_validation = true,
        .application_name = "Vulkan Gaussian update test"});
    if (!device.features().ray_tracing)
        SKIP("selected Vulkan device does not expose ray tracing");

    Scene scene;
    Gaussian gaussian{};
    gaussian.transform[0] = {0.1f, 0.0f, 0.0f};
    gaussian.transform[1] = {0.0f, 0.1f, 0.0f};
    gaussian.transform[2] = {0.0f, 0.0f, 0.1f};
    gaussian.transform[3] = {0.0f, 0.0f, 0.0f};
    gaussian.opacity = 0.8f;
    gaussian.sphericalHarmonics.count = 1;
    gaussian.setShCoefficient(0, {0.5f, 0.25f, 0.75f});
    const auto asset = scene.add(GaussianAsset(scene, "Gaussian", {gaussian}));
    auto object = std::make_unique<GaussianInstance>(
        scene, "Gaussian", asset, Transform{});
    GaussianInstance* instance = object.get();
    scene.add(std::move(object));

    VulkanScene nativeScene(device, scene);
    const gpu::AccelerationStructureHandle original = nativeScene.topLevelHandle();
    REQUIRE(original);

    instance->setPosition({0.5f, 1.0f, -2.0f});
    REQUIRE(nativeScene.updateMutableData(scene));
    CHECK(nativeScene.topLevelHandle().value == original.value);

    gaussian.opacity = 0.2f;
    gaussian.transform[3] = {0.25f, 0.0f, 0.0f};
    instance->getGaussianAsset().setGaussian(0, gaussian);
    REQUIRE(nativeScene.updateMutableData(scene));
    CHECK(nativeScene.topLevelHandle().value == original.value);
}
