#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <vector>

#include "Scene/Resources/Texture.h"

TEST_CASE("RGBA8 textures preserve byte storage", "[texture]")
{
    constexpr std::array<uint8_t, 4> pixels{128, 64, 32, 192};
    Texture texture(
        "srgb", pixels.data(), 1, 1, TextureEncoding::Srgb8);

    REQUIRE(texture.usesByteStorage());
    CHECK(texture.getEncoding() == TextureEncoding::Srgb8);
    CHECK(texture.getBytePixels()
        == std::vector<uint8_t>(pixels.begin(), pixels.end()));

    // CPU consumers still receive linear RGB, without converting alpha.
    const std::vector<float>& linear = texture.getPixels();
    REQUIRE(linear.size() == 4);
    CHECK(linear[0] == Catch::Approx(0.21586f).margin(1e-5f));
    CHECK(linear[1] == Catch::Approx(0.05127f).margin(1e-5f));
    CHECK(linear[2] == Catch::Approx(0.01444f).margin(1e-5f));
    CHECK(linear[3] == Catch::Approx(192.0f / 255.0f));
}

TEST_CASE("HDR textures retain float storage", "[texture]")
{
    constexpr std::array<float, 4> pixels{2.0f, 1.0f, 0.5f, 1.0f};
    Texture texture(
        "hdr", pixels.data(), 1, 1, TextureEncoding::Float32);

    CHECK_FALSE(texture.usesByteStorage());
    CHECK(texture.getEncoding() == TextureEncoding::Float32);
    CHECK(texture.getPixels()
        == std::vector<float>(pixels.begin(), pixels.end()));
}

TEST_CASE("Decoded texture vectors are adopted without a pixel copy",
    "[texture]")
{
    std::vector<uint8_t> pixels{1, 2, 3, 4};
    const uint8_t* decodedAddress = pixels.data();
    Texture texture(
        "adopted", std::move(pixels), 1, 1, TextureEncoding::Linear8);

    REQUIRE(texture.usesByteStorage());
    CHECK(texture.getBytePixels().data() == decodedAddress);
}

TEST_CASE("Content aliases share decoded texture storage", "[texture]")
{
    const auto pixels = std::make_shared<const std::vector<uint8_t>>(
        std::initializer_list<uint8_t>{128, 64, 32, 255});
    Texture linear(
        "linear", pixels, 1, 1, TextureEncoding::Linear8);
    Texture srgb(
        "srgb", pixels, 1, 1, TextureEncoding::Srgb8);

    CHECK(linear.getBytePixels().data() == pixels->data());
    CHECK(srgb.getBytePixels().data() == pixels->data());
    CHECK(linear.getBytePixels().data() == srgb.getBytePixels().data());
}

TEST_CASE("RGBA16F textures retain native half storage", "[texture]")
{
    // 2.0, 1.0, 0.5 and 1.0 as IEEE-754 binary16 bit patterns.
    std::vector<uint16_t> pixels{0x4000, 0x3c00, 0x3800, 0x3c00};
    const uint16_t* decodedAddress = pixels.data();
    Texture texture(
        "half", std::move(pixels), 1, 1, TextureEncoding::Float16);

    REQUIRE(texture.usesHalfStorage());
    CHECK_FALSE(texture.usesByteStorage());
    CHECK(texture.getEncoding() == TextureEncoding::Float16);
    CHECK(texture.getHalfPixels().data() == decodedAddress);

    const std::vector<float>& expanded = texture.getPixels();
    REQUIRE(expanded.size() == 4);
    CHECK(expanded[0] == Catch::Approx(2.0f));
    CHECK(expanded[1] == Catch::Approx(1.0f));
    CHECK(expanded[2] == Catch::Approx(0.5f));
    CHECK(expanded[3] == Catch::Approx(1.0f));
}
