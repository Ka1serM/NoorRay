#include <catch2/catch_test_macros.hpp>

#include "libnoorray/Scene/PbrtParser.h"

TEST_CASE("PBRT parser keeps only NoorRay-supported directives", "[pbrt]")
{
    const auto commands = nr::pbrt::parseString(R"PBRT(
        Film "rgb" "integer xresolution" 640 "integer yresolution" [480]
        PixelFilter "gaussian" "float xradius" 1.5
        Sampler "sobol" "integer pixelsamples" 64
        MakeNamedMedium "fog" "string type" "homogeneous"
        MediumInterface "fog" ""
        LookAt 0 1 -5  0 1 0  0 1 0
        Camera "perspective" "float fov" 45
        WorldBegin
        Material "diffuse" "rgb reflectance" [0.2 0.4 0.8]
        Shape "sphere" "float radius" 2
        WorldEnd
    )PBRT");

    REQUIRE(commands.size() == 7);
    CHECK(commands[0].name == "Film");
    CHECK(commands[1].name == "LookAt");
    CHECK(commands[2].name == "Camera");
    CHECK(commands[3].name == "WorldBegin");
    CHECK(commands[4].name == "Material");
    CHECK(commands[5].name == "Shape");
    CHECK(commands[6].name == "WorldEnd");
    REQUIRE(commands[0].find("xresolution"));
    CHECK(commands[0].find("xresolution")->intValue() == 640);
    CHECK(commands[4].find("reflectance")->floatValues()
        == std::vector<float>{0.2f, 0.4f, 0.8f});
}

TEST_CASE("PBRT parser handles transform matrices and escaped strings", "[pbrt]")
{
    const auto commands = nr::pbrt::parseString(R"PBRT(
        Transform [1 0 0 2  0 1 0 3  0 0 1 4  0 0 0 1]
        Texture "albedo" "spectrum" "imagemap" "string filename" "maps/a\"b.png"
    )PBRT", "fixture.pbrt");

    REQUIRE(commands.size() == 2);
    CHECK(commands[0].arguments.size() == 16);
    CHECK(commands[0].arguments[3] == "2");
    REQUIRE(commands[1].find("filename"));
    CHECK(commands[1].find("filename")->stringValue() == "maps/a\"b.png");
}
