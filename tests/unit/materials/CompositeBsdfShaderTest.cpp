// Compiles NoorRay's Slang BSDF library through the gpu compute path. This
// lives with NoorRay rather than the gpu tests because the shader under test
// is src/libnoorray/Shaders/Raytracer, not anything the RHI owns.
#include <gpu/gpu.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iterator>
#include <span>
#include <vector>

namespace {
std::vector<std::byte> read_shader(const char* path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::vector<std::byte> result(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i)
        result[i] = static_cast<std::byte>(bytes[i]);
    return result;
}

gpu::Device make_device() {
    return gpu::Device({.enable_validation = true, .application_name = "noorray bsdf test"});
}
}

TEST_CASE("gpu API executes the tagged-storage Slang BSDF composite") {
    // EnergyLutElementCount is 14,112 packed unorm16 values, two per word.
    constexpr std::size_t energy_lut_words = 14'112u / 2u;
    constexpr std::size_t spectral_table_floats =
        1'520u + 64u + 3u * 64u * 64u * 64u * 3u;
    gpu::Device device = make_device();
    auto shader = device.create_shader(read_shader(GPU_COMPOSITE_BSDF_SHADER));
    auto output = device.buffer<gpu::float4>(1);
    auto energy_luts = device.buffer<std::uint32_t>(energy_lut_words);
    auto spectral_tables = device.buffer<float>(spectral_table_floats);
    const std::vector<std::uint32_t> unit_luts(energy_lut_words, 0xFFFFFFFFu);
    energy_luts.upload(std::span<const std::uint32_t>(unit_luts));
    std::vector<float> spectral(spectral_table_floats, 0.0f);
    for (std::size_t i = 0; i < 64u; ++i)
        spectral[1'520u + i] = static_cast<float>(i) / 63.0f;
    spectral_tables.upload(std::span<const float>(spectral));

    struct Args {
        gpu::GpuPtr<gpu::float4> output;
        gpu::GpuPtr<std::uint32_t> energy_luts;
        gpu::GpuPtr<float> spectral_tables;
    } args{output.ptr(), energy_luts.ptr(), spectral_tables.ptr()};

    device.compute(shader).launch({1, 1, 1}, args);
    device.synchronize();

    gpu::float4 actual{};
    output.download(std::span<gpu::float4>(&actual, 1));
    REQUIRE(std::isfinite(actual.x));
    REQUIRE(std::isfinite(actual.y));
    REQUIRE(std::isfinite(actual.z));
    REQUIRE(actual.w >= 0.0f);
}
