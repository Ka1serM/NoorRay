#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>

class RenderTestFixture
{
protected:
    static std::string render(const char* scene, const int spp, const char* output,
        const uint32_t width = 0, const uint32_t height = 0)
    {
        const std::string path = output;
        const std::filesystem::path scenePath = std::filesystem::path(scene).is_absolute()
            ? std::filesystem::path(scene)
            : std::filesystem::path(TEST_SCENE_DIR) / scene;
        const std::string command = std::string("\"") + NOORRAY_EXECUTABLE + "\" --cli --scene \""
            + scenePath.string() + "\" --spp " + std::to_string(spp)
            + (width != 0 ? " --width " + std::to_string(width) : "")
            + (height != 0 ? " --height " + std::to_string(height) : "")
            + " --output \"" + path + "\"";
        INFO("Command: " << command);
        REQUIRE(std::system(command.c_str()) == 0);
        return path;
    }
};
