#include "NoorRaySession.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/PerspectiveCamera.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Import/PbrtParser.h"
#include "Scene/Scene.h"
#include "Scene/Import/SceneImporter.h"
#include "Scene/Import/SceneWriter.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <glm/geometric.hpp>
#include <sstream>

TEST_CASE("PBRT scene export is readable by the PBRT importer", "[scene][pbrt]")
{
    noorray::NoorRaySession session;
    Scene& scene = session.scene;
    scene.getRenderSettings().maxSamples = 17;

    const auto material = nr::materialx::documentFromAuthoring(MaterialAuthoring{});
    const auto asset = scene.add(MeshAsset::CreateSphere(scene, "ExportSphere", material, 8, 16));
    scene.add(std::make_unique<MeshInstance>(scene, "ExportSphere", asset,
        Transform({1.f, 2.f, -3.f}, {10.f, 20.f, 30.f}, {2.f, 1.f, 0.5f})));

    auto camera = std::make_unique<CameraInstance>(
        std::make_unique<PerspectiveCamera>(), "ExportCamera", Transform({0.f, 0.f, 5.f}));
    camera->getCamera()->getSensor().setResolution(64, 32);
    CameraInstance* cameraPtr = camera.get();
    scene.add(std::move(camera));
    REQUIRE(scene.setActiveCamera(cameraPtr));

    auto light = std::make_unique<LightInstance>(scene, "ExportLight",
        Transform({2.f, 3.f, 4.f}), LightInstance::TypePoint);
    light->setPhotometry({1.f, 0.5f, 0.25f}, 4.f);
    scene.add(std::move(light));

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "noorray-pbrt-export-test.pbrt";
    SceneWriter::Write(scene, path.string());
    REQUIRE(std::filesystem::is_regular_file(path));

    const auto commands = nr::pbrt::parseFile(path);
    REQUIRE(std::ranges::count_if(commands, [](const auto& command) {
        return command.name == "Shape";
    }) == 1);
    REQUIRE(std::ranges::count_if(commands, [](const auto& command) {
        return command.name == "LightSource";
    }) == 2);

    Scene reloaded;
    SceneImporter::ImportPbrtScene(reloaded, path.string());
    REQUIRE(reloaded.getMeshInstances().size() == 1);
    REQUIRE(reloaded.getActiveCamera() != nullptr);
    REQUIRE(glm::distance(reloaded.getActiveCamera()->getPosition(), glm::vec3(0.f, 0.f, 5.f)) < 1e-4f);
    REQUIRE(glm::dot(reloaded.getActiveCamera()->getRotation() * CameraInstance::LocalForward,
        glm::vec3(0.f, 0.f, -1.f)) > 0.999f);
    REQUIRE(reloaded.getPointLightCount() == 1);

    std::filesystem::remove(path);
}
