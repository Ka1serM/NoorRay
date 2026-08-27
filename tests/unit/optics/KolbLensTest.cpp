#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "Rendering/Optics/KolbLens.h"

TEST_CASE("Sellmeier and native surface geometry are self-contained", "[optics]")
{
    const auto bk7 = nr::optics::Medium{constantIorSellmeier(1.5168f)};
    REQUIRE(sellmeierIor(bk7.sellmeier, FraunhoferGreenNm) > 1.514f);
    REQUIRE(sellmeierIor(bk7.sellmeier, FraunhoferGreenNm) < 1.519f);
    nr::optics::Surface sphere{};
    sphere.radius = 50.f;
    sphere.apertureRadius = 20.f;
    REQUIRE(nr::optics::sag(sphere, 100.f) > 0.f);
    glm::vec3 hit;
    REQUIRE(nr::optics::intersect(sphere, {0.f, 0.f, -10.f}, {0.f, 0.f, 1.f}, hit));
    glm::vec3 transmitted;
    REQUIRE_FALSE(nr::optics::refract(glm::normalize(glm::vec3(.99f, 0.f, .1f)), {0.f, 0.f, -1.f}, 1.5f, 1.f, transmitted));
}

TEST_CASE("Native ZMX and AGF loaders resolve named glass", "[optics]")
{
    const auto root = std::filesystem::temp_directory_path() / "noorray-native-optics-test";
    std::filesystem::create_directories(root);
    const auto agf = root / "test.agf";
    const auto zmx = root / "test.zmx";
    {
        std::ofstream out(agf);
        out << "NM TESTGLASS 1 0 1.5 50 0 0 0\n"
               "CD 1.0 0.01 0.2 0.02 0.3 0.03\n";
    }
    {
        std::ofstream out(zmx);
        out << "SURF 0\nDISZ 10\nDIAM 10\n"
               "SURF 1\nCURV 0.02\nDISZ 5\nDIAM 10\nGLAS TESTGLASS\n"
               "SURF 2\nCURV -0.02\nDISZ 0\nDIAM 10\n"
               "SURF 3\nCURV 0\nDISZ 0\nDIAM 10\n";
    }
    const auto parsed = nr::optics::loadZmx(zmx.string(), {agf.string()});
    REQUIRE(parsed.snapshot.surfaceCount == 3);
    REQUIRE(parsed.snapshot.mediumCount >= 2);
    REQUIRE_THAT(parsed.snapshot.rearPupilRadius, Catch::Matchers::WithinAbs(10.f, 1e-5f));
    std::filesystem::remove_all(root);
}

TEST_CASE("Native AGF loader accepts UTF-16LE catalogs", "[optics]")
{
    const auto root = std::filesystem::temp_directory_path() / "noorray-native-optics-utf16-test";
    std::filesystem::create_directories(root);
    const auto agf = root / "utf16.agf";
    const auto zmx = root / "utf16.zmx";
    auto writeUtf16 = [](const std::filesystem::path& path, const std::string& ascii) {
        std::ofstream out(path, std::ios::binary);
        out.put(static_cast<char>(0xff)); out.put(static_cast<char>(0xfe));
        for (const char c : ascii) { out.put(c); out.put('\0'); }
    };
    writeUtf16(agf, "NM UTFGLASS 2 0 1.5 50 0 0 0\nCD 1.0 0.01 0.2 0.02 0.3 0.03\n");
    std::ofstream(zmx) << "SURF 0\nDISZ 10\nDIAM 10\n"
                          "SURF 1\nCURV 0.02\nDISZ 5\nDIAM 10\nGLAS UTFGLASS\n"
                          "SURF 2\nCURV -0.02\nDISZ 0\nDIAM 10\n";
    const auto parsed = nr::optics::loadZmx(zmx.string(), {agf.string()});
    REQUIRE(parsed.snapshot.mediumCount >= 2);
    std::filesystem::remove_all(root);
}

TEST_CASE("Native ZMX loader rejects unsupported commands with a source line", "[optics]")
{
    const auto root = std::filesystem::temp_directory_path() / "noorray-native-optics-invalid";
    std::filesystem::create_directories(root);
    const auto zmx = root / "invalid.zmx";
    std::ofstream(zmx) << "SURF 0\nDISZ 1\nDIAM 1\n"
                          "SURF 1\nCURV 0\nDISZ 1\nDIAM 1\n"
                          "SURF 2\nCURV 0\nDIAM 1\nUNKNOWN 7\n";
    try {
        (void)nr::optics::loadZmx(zmx.string(), {});
        FAIL("unsupported ZMX command was accepted");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()).find("invalid.zmx:") != std::string::npos);
        CHECK(std::string(error.what()).find("UNKNOWN") != std::string::npos);
    }
    std::filesystem::remove_all(root);
}

TEST_CASE("Representative Schott fixture preserves native asphere and focal metadata", "[optics][fixture]")
{
    const std::filesystem::path root = OPTICS_TEST_DIR;
    const auto parsed = nr::optics::loadZmx(
        (root / "kolb_doublet.zmx").string(),
        {(root / "schott_fixture.agf").string()});
    REQUIRE(parsed.snapshot.surfaceCount == 5);
    CHECK(parsed.snapshot.focalLengthMm > 1.0f);
    CHECK(parsed.snapshot.surfaces[3].conic == Catch::Approx(-0.2f));
    CHECK(parsed.snapshot.surfaces[3].asphere[0] == Catch::Approx(1.0e-6f));
    CHECK(parsed.snapshot.rearPupilRadius > 0.0f);
}
