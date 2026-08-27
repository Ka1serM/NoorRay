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

// Every test runs with the validation layers on, so a spec violation fails the
// build rather than going unnoticed.
gpu::Device make_device() {
    return gpu::Device({.enable_validation = true, .application_name = "gpu tests"});
}
}

TEST_CASE("gpu API adds typed buffers through a root argument") {
    constexpr std::size_t count = 257;

    gpu::Device device = make_device();
    auto shader = device.create_shader(read_shader(GPU_TEST_SHADER));
    auto a = device.buffer<float>(count);
    auto b = device.buffer<float>(count);
    auto output = device.buffer<float>(count);
    const auto a_resource = a.handle();
    REQUIRE(a_resource);
    REQUIRE(a.handle().value == a_resource.value);

    std::vector<float> lhs(count), rhs(count), expected(count);
    for (std::size_t i = 0; i < count; ++i) {
        lhs[i] = static_cast<float>(i) * 0.25f;
        rhs[i] = 3.0f - static_cast<float>(i) * 0.125f;
        expected[i] = lhs[i] + rhs[i];
    }
    device.upload(a, std::span<const float>(lhs));
    device.upload(b, std::span<const float>(rhs));

    struct Args {
        gpu::GpuPtr<float> a;
        gpu::GpuPtr<float> b;
        gpu::GpuPtr<float> output;
        std::uint32_t count;
    } args{a.ptr(), b.ptr(), output.ptr(), static_cast<std::uint32_t>(count)};

    // The temporary pipeline is destroyed immediately after launch; the
    // submission must retain the Vulkan pipeline until GPU completion.
    device.compute(shader).launch(
        {static_cast<std::uint32_t>((count + 63) / 64), 1, 1}, args);
    device.barrier(gpu::Stage::Compute, gpu::Stage::Copy);
    const auto token = device.signal();
    device.wait(token);

    std::vector<float> actual(count);
    device.download(std::span<float>(actual), output);
    REQUIRE(actual == expected);
}

TEST_CASE("gpu API dispatches compute indirectly with root arguments") {
    constexpr std::size_t count = 64;
    gpu::Device device = make_device();
    auto shader = device.create_shader(read_shader(GPU_TEST_SHADER));
    auto pipeline = device.compute(shader);

    auto a = device.buffer<float>(count);
    auto b = device.buffer<float>(count);
    auto output = device.buffer<float>(count);
    const std::vector<float> ones(count, 1.0f);
    const std::vector<float> twos(count, 2.0f);
    device.upload(a, std::span<const float>(ones));
    device.upload(b, std::span<const float>(twos));

    auto groups = device.buffer<gpu::DispatchArgs>(1);
    const gpu::DispatchArgs dispatch{1, 1, 1};
    device.upload(groups, std::span<const gpu::DispatchArgs>(&dispatch, 1));

    struct Args {
        gpu::GpuPtr<float> a;
        gpu::GpuPtr<float> b;
        gpu::GpuPtr<float> output;
        std::uint32_t count;
    } args{a.ptr(), b.ptr(), output.ptr(), static_cast<std::uint32_t>(count)};

    pipeline.launch_indirect(groups.ptr(), args);
    device.synchronize();

    std::vector<float> actual(count);
    device.download(std::span<float>(actual), output);
    REQUIRE(actual == std::vector<float>(count, 3.0f));
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
    device.upload(energy_luts, std::span<const std::uint32_t>(unit_luts));
    std::vector<float> spectral(spectral_table_floats, 0.0f);
    for (std::size_t i = 0; i < 64u; ++i)
        spectral[1'520u + i] = static_cast<float>(i) / 63.0f;
    device.upload(spectral_tables, std::span<const float>(spectral));

    struct Args {
        std::uint32_t output;
        std::uint32_t energy_luts;
        std::uint32_t spectral_tables;
    } args{
        static_cast<std::uint32_t>(output.handle().value),
        static_cast<std::uint32_t>(energy_luts.handle().value),
        static_cast<std::uint32_t>(spectral_tables.handle().value),
    };

    device.compute(shader).launch({1, 1, 1}, args);
    device.synchronize();

    gpu::float4 actual{};
    device.download(std::span<gpu::float4>(&actual, 1), output);
    REQUIRE(std::isfinite(actual.x));
    REQUIRE(std::isfinite(actual.y));
    REQUIRE(std::isfinite(actual.z));
    REQUIRE(actual.w >= 0.0f);
}

TEST_CASE("gpu API round-trips image contents") {
    gpu::Device device = make_device();
    auto image = device.image<std::uint8_t>(4, 4,
        gpu::ImageUsage::Storage | gpu::ImageUsage::Sampled);
    REQUIRE(image.storage_handle());
    REQUIRE(image.sampled_handle());
    REQUIRE(image.storage_handle().value != image.sampled_handle().value);

    std::vector<std::uint8_t> pixels(4u * 4u * 4u);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<std::uint8_t>(i * 3u + 1u);
    device.upload(image, std::span<const std::uint8_t>(pixels));

    std::vector<std::uint8_t> read_back(pixels.size());
    device.download(std::span<std::uint8_t>(read_back), image);
    REQUIRE(read_back == pixels);
}

TEST_CASE("gpu API rasterizes a triangle into a render target") {
    gpu::Device device = make_device();
    const auto triangle = read_shader(GPU_TRIANGLE_SHADER);
    auto vertex = device.create_shader(triangle, "vertMain");
    auto fragment = device.create_shader(triangle, "fragMain");

    constexpr std::uint32_t size = 64;
    auto target = device.image<std::uint8_t>(size, size,
        gpu::ImageUsage::ColorAttachment, gpu::ImageFormat::Rgba8Unorm);
    auto pipeline = device.graphics({vertex, fragment, {}, gpu::ImageFormat::Rgba8Unorm});

    device.render({target.handle(), {}}, [&] { pipeline.draw(3); });
    device.synchronize();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size * 4u);
    device.download(std::span<std::uint8_t>(pixels), target);
    const auto texel = [&](const std::uint32_t x, const std::uint32_t y) {
        return std::span<const std::uint8_t>(
            pixels.data() + (static_cast<std::size_t>(y) * size + x) * 4u, 4u);
    };
    // The triangle covers the middle of the target and none of the corners.
    const auto centre = texel(size / 2, size / 2);
    REQUIRE((centre[0] || centre[1] || centre[2]));
    const auto corner = texel(0, 0);
    REQUIRE(corner[0] == 0);
    REQUIRE(corner[1] == 0);
    REQUIRE(corner[2] == 0);

    // A pipeline built for a different attachment format must be rejected
    // rather than silently mismatching the render pass instance.
    auto bgra_target = device.image<std::uint8_t>(8, 8,
        gpu::ImageUsage::ColorAttachment, gpu::ImageFormat::Bgra8Unorm);
    REQUIRE_THROWS_AS(
        device.render({bgra_target.handle(), {}}, [&] { pipeline.draw(3); }), gpu::Error);
}

TEST_CASE("gpu API draws indirectly and honours depth targets") {
    gpu::Device device = make_device();
    const auto triangle = read_shader(GPU_TRIANGLE_SHADER);
    auto vertex = device.create_shader(triangle, "vertMain");
    auto fragment = device.create_shader(triangle, "fragMain");

    constexpr std::uint32_t size = 32;
    auto target = device.image<std::uint8_t>(size, size, gpu::ImageUsage::ColorAttachment);
    auto depth = device.image<float>(size, size, gpu::ImageUsage::DepthAttachment);

    gpu::GraphicsState state{};
    state.depth_test = true;
    state.depth_write = true;
    auto depth_pipeline = device.graphics({vertex, fragment, state});
    auto plain_pipeline = device.graphics({vertex, fragment, {}});

    auto commands = device.buffer<gpu::DrawArgs>(1);
    const gpu::DrawArgs draw_args{3, 1, 0, 0};
    device.upload(commands, std::span<const gpu::DrawArgs>(&draw_args, 1));

    device.render({target.handle(), depth.handle()},
        [&] { depth_pipeline.draw_indirect(commands.ptr()); });
    device.synchronize();

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size * 4u);
    device.download(std::span<std::uint8_t>(pixels), target);
    const std::size_t centre =
        (static_cast<std::size_t>(size / 2) * size + size / 2) * 4u;
    REQUIRE((pixels[centre] || pixels[centre + 1] || pixels[centre + 2]));

    // Depth attachment presence is part of the pipeline's contract.
    REQUIRE_THROWS_AS(device.render({target.handle(), depth.handle()},
        [&] { plain_pipeline.draw(3); }), gpu::Error);
    REQUIRE_THROWS_AS(device.render({target.handle(), {}},
        [&] { depth_pipeline.draw(3); }), gpu::Error);
}

TEST_CASE("gpu API exposes samplers and rejects invalid resources") {
    gpu::Device device = make_device();
    auto sampler = device.sampler({});
    REQUIRE(sampler.handle());

    REQUIRE_THROWS_AS(device.buffer<float>(0), gpu::Error);
    REQUIRE_THROWS_AS(device.image<std::uint8_t>(0, 4, gpu::ImageUsage::Storage), gpu::Error);
    REQUIRE_THROWS_AS(gpu::Buffer<float>{}.ptr(), gpu::Error);
    REQUIRE_THROWS_AS(gpu::ComputePipeline{}.launch({1, 1, 1}, 0u), gpu::Error);
}

TEST_CASE("gpu API reclaims resource descriptors after GPU retirement") {
    gpu::Device device = make_device();
    // Exceed the complete resource heap several times while keeping only one
    // descriptor live. This models camera and immutable scene snapshots being
    // replaced during a long editor session.
    for (std::uint32_t i = 0; i < 12'000u; ++i) {
        {
            auto snapshot = device.buffer<std::uint32_t>(1);
            REQUIRE(snapshot.handle());
        }
        if ((i & 31u) == 31u) {
            const auto token = device.signal();
            device.wait(token);
        }
    }
    device.synchronize();
}
