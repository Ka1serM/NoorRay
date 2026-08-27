#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include "UI/NoorRayUi.h"
#include "Log.h"
#include "Backend/Vulkan/Raytracer/RaytracerRenderer.h"
#include "Backend/Vulkan/Raytracer/CameraSnapshot.h"
#include <gpu/gpu.hpp>
#include "Materials/MaterialX/MaterialXSceneRuntime.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/RealisticCamera.h"
#include "Rendering/Camera/ThinLensCamera.h"
#include "Rendering/Camera/FisheyeCamera.h"
#include "Scene/Scene.h"
#include "IO/BitmapWriter.h"
#include "IO/Bitmap.h"

namespace
{

Bitmap readColor(VulkanRaytracer& renderer)
{
    const auto beauty = renderer.readBeauty();
    std::vector<glm::vec4> pixels(static_cast<std::size_t>(renderer.width())
        * renderer.height());
    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        pixels[i] = {beauty[i].x, beauty[i].y, beauty[i].z, beauty[i].w};
    }
    return Bitmap(renderer.width(), renderer.height(), std::move(pixels));
}

void uploadActiveCamera(VulkanRaytracer& renderer, const Scene& scene)
{
    VulkanCameraSnapshot snapshot{};
    nr::optics::LensSnapshot lens{};
    if (const CameraInstance* instance = scene.getRenderCamera())
    {
        const Camera* camera = instance->getCamera();
        for (uint32_t row = 0; row < 4; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                snapshot.cameraToWorld[row * 4u + column]
                    = camera->cameraToWorld[column][row];
        snapshot.projection = static_cast<uint32_t>(instance->getProjectionType());
        snapshot.sensorWidthMm = camera->getSensor().filmWidth();
        snapshot.sensorHeightMm = camera->getSensor().filmHeight();
        snapshot.focalLengthMm = camera->getFocalLengthMm();
        snapshot.focusDistanceCm = camera->getFocusDistanceCm();
        snapshot.sensorOrigin = static_cast<uint32_t>(
            camera->getSensor().origin());
        snapshot.exposure = camera->exposure;
        if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>())
        {
            snapshot.apertureDiameterMm = realistic->apertureDiameterMm;
            lens = realistic->optics;
        }
        else if (const auto* thinLens = camera->CastOrNullptr<ThinLensCamera>())
            snapshot.apertureDiameterMm = thinLens->apertureDiameterMm;
        else if (const auto* fisheye = camera->CastOrNullptr<FisheyeCamera>())
            snapshot.apertureDiameterMm = fisheye->apertureDiameterMm;
    }
    renderer.uploadLensSnapshot(lens);
    renderer.uploadCameraSnapshot(snapshot);
}

void uploadCompiledMaterials(VulkanRaytracer& renderer, Scene& scene,
    const std::string& scenePath)
{
    MaterialXSceneRuntime materialRuntime;
    const std::string sceneDirectory =
        std::filesystem::path(scenePath).parent_path().string();
    do
    {
        materialRuntime.compilePending(scene, sceneDirectory);
        if (materialRuntime.hasPendingCompilations())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (materialRuntime.hasPendingCompilations());
    renderer.uploadMaterials(scene, materialRuntime.programs().words(),
        materialRuntime.programs().textureIndices());
    renderer.uploadEnvironment(scene);
}

struct CliOptions
{
    std::string scenePath;
    std::string outputPath{"output.exr"};
    int samplesPerPixel{64};
    uint32_t sampleSeed{};
    int maxBounces{-1};
    int width{};
    int height{};
    int windowWidth{};
    int windowHeight{};
    std::optional<GaussianShadingMode> gaussianShadingMode;
    std::optional<GaussianProxyType> gaussianProxyType;
    bool aovEnabled{};
    bool statsEnabled{};
    bool cliMode{};
    bool vulkanRaytracerSmoke{};
    bool showHelp{};
};

void printUsage()
{
    std::cerr << "Usage:\n"
        << "  NoorRay                                                     # GUI mode\n"
        << "  NoorRay --scene <file>                                      # Open a scene in the GUI\n"
        << "  NoorRay --cli --scene <file> --spp <N> [options]            # Render headlessly\n"
        << "\nOptions:\n"
        << "  --cli                Render headlessly instead of opening the GUI\n"
        << "  --output <path>      Output image path (default: output.exr)\n"
        << "  --seed <uint>        Owen-scramble seed for independent renders\n"
        << "  --width  <int>       Output width (default: from camera sensor or 1280)\n"
        << "  --height <int>       Output height (default: from camera sensor or 720)\n"
        << "  --window-width <int>  GUI window width (default: auto, 2/3 of display)\n"
        << "  --window-height <int> GUI window height (default: auto, 2/3 of display)\n"
        << "  --max-bounces <int>  Maximum path depth (default: from scene)\n"
        << "  --gaussian-shading <direct|gi>\n"
        << "                       Override the scene's Gaussian shading mode\n"
        << "  --gaussian-proxy <icosphere|octahedron|icosahedron|icosphere2>\n"
        << "                       Override the Gaussian tracing proxy geometry\n"
        << "  --aov                Render surface AOVs (for profiling or diagnostics)\n"
        << "  --stats              Print a per-kernel GPU timing breakdown after rendering\n"
        << "  --vulkan-raytracer-smoke  Run one native Vulkan raytracer dispatch and save it\n";
}

const char* requireValue(const int argc, char* argv[], int& index)
{
    if (++index >= argc)
        throw std::invalid_argument(std::string("Missing value after ") + argv[index - 1]);
    return argv[index];
}

CliOptions parseOptions(const int argc, char* argv[])
{
    CliOptions options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
            options.showHelp = true;
        else if (arg == "--cli")
            options.cliMode = true;
        else if (arg == "--vulkan-raytracer-smoke")
        {
            options.vulkanRaytracerSmoke = true;
            options.cliMode = true;
        }
        else if (arg == "--scene" || arg == "--import" || arg == "--ply")
            options.scenePath = requireValue(argc, argv, i);
        else if (arg == "--spp")
            options.samplesPerPixel = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--seed")
            options.sampleSeed = static_cast<uint32_t>(std::stoul(requireValue(argc, argv, i)));
        else if (arg == "--output")
            options.outputPath = requireValue(argc, argv, i);
        else if (arg == "--width")
            options.width = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--height")
            options.height = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--window-width")
            options.windowWidth = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--window-height")
            options.windowHeight = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--max-bounces")
            options.maxBounces = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--gaussian-shading")
        {
            const std::string mode = requireValue(argc, argv, i);
            if (mode == "direct")
                options.gaussianShadingMode = GaussianShadingMode::DirectColor;
            else if (mode == "gi")
                options.gaussianShadingMode = GaussianShadingMode::GlobalIllumination;
            else
                throw std::invalid_argument(
                    "--gaussian-shading must be either 'direct' or 'gi'");
        }
        else if (arg == "--gaussian-proxy")
        {
            const std::string type = requireValue(argc, argv, i);
            if (type == "icosphere")
                options.gaussianProxyType = GaussianProxyType::Icosphere;
            else if (type == "octahedron")
                options.gaussianProxyType = GaussianProxyType::Octahedron;
            else if (type == "icosahedron")
                options.gaussianProxyType = GaussianProxyType::Icosahedron;
            else if (type == "icosphere2")
                options.gaussianProxyType = GaussianProxyType::IcosphereLevel2;
            else
                throw std::invalid_argument(
                    "--gaussian-proxy must be 'icosphere', 'octahedron', 'icosahedron', or 'icosphere2'");
        }
        else if (arg == "--stats")
            options.statsEnabled = true;
        else if (arg == "--aov")
            options.aovEnabled = true;
        else if (arg.starts_with("--"))
            throw std::invalid_argument("Unknown flag: " + arg);
        else
            throw std::invalid_argument("Unexpected positional argument: " + arg);
    }

    if (!options.showHelp && options.scenePath.empty() && !options.vulkanRaytracerSmoke)
        throw std::invalid_argument("No scene file specified. Use --scene.");
    if (options.samplesPerPixel < 1)
        throw std::invalid_argument("--spp must be greater than zero");
    if (options.width < 0 || options.height < 0)
        throw std::invalid_argument("--width and --height cannot be negative");
    if (options.windowWidth < 0 || options.windowHeight < 0)
        throw std::invalid_argument("--window-width and --window-height cannot be negative");
    if ((options.windowWidth == 0) != (options.windowHeight == 0))
        throw std::invalid_argument("--window-width and --window-height must be used together");
    if (options.maxBounces == 0 || options.maxBounces < -1)
        throw std::invalid_argument("--max-bounces must be greater than zero");
    return options;
}

void runCli(const CliOptions& options)
{
    if (options.vulkanRaytracerSmoke)
    {
        const uint32_t width = options.width > 0 ? static_cast<uint32_t>(options.width) : 128u;
        const uint32_t height = options.height > 0 ? static_cast<uint32_t>(options.height) : 72u;
        gpu::Device device;
        // The scene-less smoke mode uses the tiny native triangle scene to
        // validate the AS/query layer before a full imported scene is added.
        VulkanRaytracer renderer(device, width, height,
            options.scenePath.empty());
        std::unique_ptr<Scene> nativeScene;
        if (!options.scenePath.empty())
        {
            nativeScene = std::make_unique<Scene>();
            nativeScene->load(options.scenePath);
            if (options.gaussianShadingMode)
                nativeScene->getRenderSettings().gaussianShadingMode =
                    *options.gaussianShadingMode;
            if (options.gaussianProxyType)
                nativeScene->getRenderSettings().gaussianProxyType =
                    *options.gaussianProxyType;
            renderer.uploadScene(*nativeScene);
            uploadCompiledMaterials(renderer, *nativeScene, options.scenePath);
            uploadActiveCamera(renderer, *nativeScene);
        }
        LOG_INFO("Vulkan raytracer smoke: recording dispatch");
        renderer.render(0, 0);
        LOG_INFO("Vulkan raytracer smoke: waiting for dispatch");
        device.synchronize();

        LOG_INFO("Vulkan raytracer smoke: reading back image");
        const Bitmap bitmap = readColor(renderer);
        std::string error;
        if (!BitmapWriter::write(options.outputPath, bitmap, {}, &error))
            throw std::runtime_error("Failed to save Vulkan raytracer smoke image: " + error);
        LOG_INFO("Saved Vulkan raytracer smoke image: " << options.outputPath);
        return;
    }
    Scene scene;
    scene.load(options.scenePath);
    if (options.gaussianShadingMode)
        scene.getRenderSettings().gaussianShadingMode = *options.gaussianShadingMode;
    if (options.gaussianProxyType)
        scene.getRenderSettings().gaussianProxyType = *options.gaussianProxyType;
    if (options.maxBounces > 0)
        scene.getRenderSettings().maxBounces = options.maxBounces;
    const auto* cameraInstance = scene.getRenderCamera();
    const glm::uvec2 sceneResolution = cameraInstance
        ? cameraInstance->getCamera()->getSensor().resolution()
        : glm::uvec2(1280u, 720u);
    const uint32_t width = options.width > 0
        ? static_cast<uint32_t>(options.width) : sceneResolution.x;
    const uint32_t height = options.height > 0
        ? static_cast<uint32_t>(options.height) : sceneResolution.y;
    gpu::Device device;
    VulkanRaytracer renderer(device, width, height, false);
    renderer.uploadScene(scene);
    uploadCompiledMaterials(renderer, scene, options.scenePath);
    uploadActiveCamera(renderer, scene);
    double rayTracingMilliseconds = 0.0;
    double minimumDispatchMilliseconds = std::numeric_limits<double>::max();
    double maximumDispatchMilliseconds = 0.0;
    for (uint32_t sample = 0; sample < static_cast<uint32_t>(options.samplesPerPixel); ++sample)
    {
        renderer.render(options.sampleSeed, sample);
        device.synchronize();
        if (options.statsEnabled)
        {
            const double milliseconds = renderer.lastDispatchMilliseconds();
            rayTracingMilliseconds += milliseconds;
            minimumDispatchMilliseconds = std::min(
                minimumDispatchMilliseconds, milliseconds);
            maximumDispatchMilliseconds = std::max(
                maximumDispatchMilliseconds, milliseconds);
        }
    }
    if (options.statsEnabled)
    {
        const double samples = static_cast<double>(options.samplesPerPixel);
        const double average = samples > 0.0 ? rayTracingMilliseconds / samples : 0.0;
        std::cout << std::fixed << std::setprecision(3)
            << "GPU timing (timestamp queries):\n"
            << "  ray tracing: " << rayTracingMilliseconds << " ms total, "
            << average << " ms/sample, " << minimumDispatchMilliseconds
            << " ms min, " << maximumDispatchMilliseconds << " ms max\n";
    }
    const Bitmap bitmap = readColor(renderer);
    std::string error;
    if (!BitmapWriter::write(options.outputPath, bitmap, {}, &error))
        throw std::runtime_error("Failed to save Vulkan render: " + error);
    LOG_INFO("Saved: " << options.outputPath);
}

}

int main(const int argc, char* argv[])
{
    try
    {
        const CliOptions options = argc == 1 ? CliOptions{} : parseOptions(argc, argv);
        if (options.showHelp)
        {
            printUsage();
            return 0;
        }
        if (options.cliMode)
            runCli(options);
        else
        {
            NoorRayUi ui(options.scenePath,
                static_cast<uint32_t>(options.windowWidth),
                static_cast<uint32_t>(options.windowHeight));
            ui.run();
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(e.what());
        printUsage();
        return 1;
    }
}
