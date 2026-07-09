#include <iostream>
#include <string>
#include <cstring>
#include "NoorRay.h"
#include "Log.h"

static void printUsage()
{
    std::cerr << "Usage:\n"
        << "  NoorRay                                                     # GUI mode\n"
        << "  NoorRay --scene <file> --spp <N> [options]                  # Render scene, mesh, or Gaussian splat\n"
        << "\nOptions:\n"
        << "  --output <path>      Output image path (default: output.exr)\n"
        << "  --width  <int>       Output width (default: from camera sensor or 1280)\n"
        << "  --height <int>       Output height (default: from camera sensor or 720)\n"
        << "  --stats              Print a per-kernel GPU timing breakdown after rendering\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        NoorRay app(1680, 960);
        app.runUi();
        return 0;
    }

    std::string scenePath;
    std::string outputPath = "output.exr";
    int spp = 64;
    int width = 0, height = 0;
    bool statsEnabled = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage();
            return 0;
        }
        else if (arg == "--scene" || arg == "--import" || arg == "--ply")
        {
            if (i + 1 >= argc)
            {
                LOG_ERROR("Missing path after " << arg);
                return 1;
            }
            scenePath = argv[++i];
        }
        else if (arg == "--spp")
        {
            if (i + 1 >= argc)
            {
                LOG_ERROR("Missing value after --spp");
                return 1;
            }
            spp = std::stoi(argv[++i]);
        }
        else if (arg == "--output")
        {
            if (i + 1 >= argc)
            {
                LOG_ERROR("Missing path after --output");
                return 1;
            }
            outputPath = argv[++i];
        }
        else if (arg == "--width")
        {
            if (i + 1 >= argc)
            {
                LOG_ERROR("Missing value after --width");
                return 1;
            }
            width = std::stoi(argv[++i]);
        }
        else if (arg == "--height")
        {
            if (i + 1 >= argc)
            {
                LOG_ERROR("Missing value after --height");
                return 1;
            }
            height = std::stoi(argv[++i]);
        }
        else if (arg == "--stats")
        {
            statsEnabled = true;
        }
        else if (arg.find("--") == 0)
        {
            LOG_ERROR("Unknown flag: " << arg);
            printUsage();
            return 1;
        }
        else
        {
            LOG_ERROR("Unexpected positional argument: " << arg);
            printUsage();
            return 1;
        }
    }

    if (scenePath.empty())
    {
        LOG_ERROR("No scene file specified. Use --scene.");
        printUsage();
        return 1;
    }

    try
    {
        NoorRay app(scenePath, spp, outputPath, width, height, statsEnabled);
        app.runCli();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
