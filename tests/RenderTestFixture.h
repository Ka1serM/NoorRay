#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

class RenderTestFixture
{
protected:
    static std::string render(const char* scene, const int spp, const char* output)
    {
        const std::string path = output;
        const std::string command = std::string("\"") + NOORRAY_EXECUTABLE + "\" --scene \""
            + TEST_SCENE_DIR "/" + scene + "\" --spp " + std::to_string(spp)
            + " --output \"" + path + "\"";
        INFO("Command: " << command);
        REQUIRE(std::system(command.c_str()) == 0);
        return path;
    }
};
