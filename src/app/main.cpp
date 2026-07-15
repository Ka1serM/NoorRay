#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "NoorRaySession.h"
#include "UI/NoorRayUi.h"
#include "Camera/CameraInstance.h"
#include "Log.h"
#include "Raytracing/Raytracer.h"

namespace
{

struct CliOptions
{
    std::string scenePath;
    std::string outputPath{"output.exr"};
    int samplesPerPixel{64};
    int width{};
    int height{};
    bool statsEnabled{};
    bool denoiserEnabled{};
    bool cliMode{};
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
        << "  --width  <int>       Output width (default: from camera sensor or 1280)\n"
        << "  --height <int>       Output height (default: from camera sensor or 720)\n"
        << "  --denoise            Apply the OptiX HDR beauty denoiser\n"
        << "  --stats              Print a per-kernel GPU timing breakdown after rendering\n";
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
        else if (arg == "--scene" || arg == "--import" || arg == "--ply")
            options.scenePath = requireValue(argc, argv, i);
        else if (arg == "--spp")
            options.samplesPerPixel = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--output")
            options.outputPath = requireValue(argc, argv, i);
        else if (arg == "--width")
            options.width = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--height")
            options.height = std::stoi(requireValue(argc, argv, i));
        else if (arg == "--stats")
            options.statsEnabled = true;
        else if (arg == "--denoise")
            options.denoiserEnabled = true;
        else if (arg.starts_with("--"))
            throw std::invalid_argument("Unknown flag: " + arg);
        else
            throw std::invalid_argument("Unexpected positional argument: " + arg);
    }

    if (!options.showHelp && options.scenePath.empty())
        throw std::invalid_argument("No scene file specified. Use --scene.");
    if (options.samplesPerPixel < 1)
        throw std::invalid_argument("--spp must be greater than zero");
    if (options.width < 0 || options.height < 0)
        throw std::invalid_argument("--width and --height cannot be negative");
    return options;
}

void runCli(const CliOptions& options)
{
    noorray::NoorRaySession session;
    session.scene.load(options.scenePath);
    if (!session.scene.getActiveCamera())
    {
        auto camera = std::make_unique<PerspectiveCamera>();
        auto instance = std::make_unique<CameraInstance>(std::move(camera), "Camera",
            Transform{glm::vec3(0.0f, 2.0f, 5.0f)});
        session.scene.add(std::move(instance));
    }
    if (auto* camera = session.scene.getActiveCamera();
        camera && (options.width > 0 || options.height > 0))
    {
        Sensor& sensor = camera->getCamera()->getSensor();
        const glm::uvec2 resolution = sensor.resolution();
        sensor.setResolution(
            options.width > 0 ? static_cast<uint32_t>(options.width) : resolution.x,
            options.height > 0 ? static_cast<uint32_t>(options.height) : resolution.y);
    }

    Raytracer& raytracer = *session.raytracer;
    raytracer.setAovEnabled(false);
    raytracer.setStatsEnabled(options.statsEnabled);
    raytracer.setTimingEnabled(options.statsEnabled);
    session.scene.getRenderSettings().samples = options.samplesPerPixel;
    session.scene.getRenderSettings().optixDenoiserEnabled = options.denoiserEnabled;
    LOG_INFO("Rendering @ " << options.samplesPerPixel << " spp");
    raytracer.renderFrame(0, 0);
    raytracer.getOutputColor().save(options.outputPath);
    raytracer.harvestKernelStats();
    LOG_INFO("Saved: " << options.outputPath);
    if (options.statsEnabled)
        raytracer.printKernelStats();
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
            NoorRayUi ui(options.scenePath);
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
