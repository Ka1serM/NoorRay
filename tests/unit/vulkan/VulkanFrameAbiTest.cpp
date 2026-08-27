#include <cstddef>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "Backend/Vulkan/Raytracer/FrameAbi.h"
#include "Backend/Vulkan/Raytracer/LightSnapshot.h"

TEST_CASE("Vulkan raytracer FrameParams has a stable push-constant ABI", "[vulkan][abi]")
{
    using nr::vulkan::FrameParams;
    CHECK(offsetof(FrameParams, sceneAddress) == 0);
    CHECK(offsetof(FrameParams, colorImage) == 32);
    CHECK(offsetof(FrameParams, width) == 64);
    CHECK(offsetof(FrameParams, frameIndex) == 72);
    CHECK(offsetof(FrameParams, gaussianRecords) == 96);
    CHECK(offsetof(FrameParams, environmentBuffer) == 128);
    CHECK(offsetof(FrameParams, energyLutBuffer) == 132);
    CHECK(offsetof(FrameParams, spectralTables) == 136);
    CHECK(sizeof(FrameParams) == 168);
}

TEST_CASE("FrameParams uses explicit GPU pointer and heap-index widths", "[vulkan][abi]")
{
    using nr::vulkan::FrameParams;
    CHECK(sizeof(decltype(FrameParams::sceneAddress)) == sizeof(uint64_t));
    CHECK(sizeof(decltype(FrameParams::colorImage)) == sizeof(uint32_t));
    CHECK(sizeof(decltype(FrameParams::topLevelAS)) == sizeof(uint32_t));
}

TEST_CASE("Vulkan analytic light snapshot is scalar and bounded", "[vulkan][abi]")
{
    CHECK(sizeof(VulkanLightHeader) == 16);
    CHECK(sizeof(VulkanLightRecord) == 96);
    CHECK(offsetof(VulkanLightRecord, direction) == 16);
    CHECK(offsetof(VulkanLightRecord, color) == 32);
    CHECK(offsetof(VulkanLightRecord, params) == 64);
}
