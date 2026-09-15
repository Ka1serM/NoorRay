#include <gpu/gpu.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <span>
#include <algorithm>
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
}

TEST_CASE("gpu API traces rays against a built acceleration structure") {
    gpu::Device device({.enable_validation = true, .application_name = "gpu tests"});
    if (!device.features().ray_tracing)
        SKIP("selected Vulkan device does not expose ray tracing");

    // A triangle covering the middle of the z=0 plane, large enough that the
    // centre of the launch grid hits it and the corners miss.
    auto positions = device.buffer<gpu::float3>(3);
    auto indices = device.buffer<std::uint32_t>(3);
    const std::vector<gpu::float3> vertices{
        {-0.6f, -0.6f, 0.0f}, {0.6f, -0.6f, 0.0f}, {0.0f, 0.6f, 0.0f}};
    positions.upload(std::span<const gpu::float3>(vertices));
    indices.upload(std::span<const std::uint32_t>(
        std::vector<std::uint32_t>{0, 1, 2}));

    const gpu::TriangleGeometry triangles{positions.ptr(), indices.ptr(), 1};
    auto blas = device.build_blas(std::span<const gpu::TriangleGeometry>(&triangles, 1));
    REQUIRE(blas);

    gpu::Instance instance{blas, {}};
    for (std::size_t i = 0; i < 4; ++i)
        instance.transform.values[i][i] = 1.0f;
    auto tlas = device.build_tlas(std::span<const gpu::Instance>(&instance, 1));
    REQUIRE(tlas);
    REQUIRE(tlas.handle());

    const auto shaders = read_shader(GPU_RAYTRACING_SHADER);
    auto raygen = device.create_shader(shaders, "rayGenMain");
    auto miss = device.create_shader(shaders, "missMain");
    auto closest_hit = device.create_shader(shaders, "closestHitMain");
    auto pipeline = device.ray_tracing({raygen, {miss}, {closest_hit}, {}, {}});

    constexpr std::uint32_t width = 16;
    constexpr std::uint32_t height = 16;
    auto output = device.buffer<std::uint32_t>(width * height);

    struct Args {
        gpu::GpuPtr<std::uint32_t> result;
        std::uint64_t scene;
        std::uint32_t width;
        std::uint32_t padding = 0;
    } args{output.ptr(), tlas.handle().value, width};
    static_assert(sizeof(Args) == 24);
    static_assert(offsetof(Args, scene) == 8);
    static_assert(offsetof(Args, width) == 16);

    pipeline.trace({width, height, 1}, args);
    device.synchronize();

    std::vector<std::uint32_t> hits(width * height);
    output.download(std::span<std::uint32_t>(hits));

    // The centre of the grid lands inside the triangle; the corners do not.
    REQUIRE(hits[(height / 2) * width + width / 2] == 1u);
    REQUIRE(hits[0] == 0u);
    REQUIRE(hits[width - 1] == 0u);
    REQUIRE(hits[(height - 1) * width] == 0u);
    // Both the miss and the closest-hit group must have been reached.
    const auto hit_count = std::count(hits.begin(), hits.end(), 1u);
    REQUIRE(hit_count > 0);
    REQUIRE(hit_count < static_cast<long>(hits.size()));

    SECTION("TLAS instances can be updated in place") {
        instance.transform.values[0][3] = 10.0f;
        device.update_tlas(tlas, std::span<const gpu::Instance>(&instance, 1));
        pipeline.trace({width, height, 1}, args);
        device.synchronize();
        output.download(std::span<std::uint32_t>(hits));
        REQUIRE(std::count(hits.begin(), hits.end(), 1u) == 0);

        instance.transform.values[0][3] = 0.0f;
        device.update_tlas(tlas, std::span<const gpu::Instance>(&instance, 1));
        pipeline.trace({width, height, 1}, args);
        device.synchronize();
        output.download(std::span<std::uint32_t>(hits));
        REQUIRE(hits[(height / 2) * width + width / 2] == 1u);
    }
}
