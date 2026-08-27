#include <gpu/gpu.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace {
std::vector<std::byte> read_shader(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open shader: " + std::string(path));
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::vector<std::byte> result(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i)
        result[i] = static_cast<std::byte>(bytes[i]);
    return result;
}
}

int main() {
    try {
        gpu::Device device;
        if (!device.features().ray_tracing) {
            std::cout << "ray-tracing example skipped: feature unavailable\n";
            return 0;
        }

        const std::array<gpu::float3, 3> vertices{
            gpu::float3{-1.0f, -1.0f, 0.0f},
            gpu::float3{1.0f, -1.0f, 0.0f},
            gpu::float3{0.0f, 1.0f, 0.0f}};
        const std::array<std::uint32_t, 3> indices{0, 1, 2};
        auto positions = device.buffer<gpu::float3>(vertices.size());
        auto index_buffer = device.buffer<std::uint32_t>(indices.size());
        device.upload(positions, std::span<const gpu::float3>(vertices));
        device.upload(index_buffer, std::span<const std::uint32_t>(indices));

        const gpu::TriangleGeometry geometry{positions.ptr(), index_buffer.ptr(), 1};
        const auto blas = device.build_blas(std::span<const gpu::TriangleGeometry>(&geometry, 1));
        gpu::Instance instance{blas, {}};
        instance.transform.values[0][0] = 1.0f;
        instance.transform.values[1][1] = 1.0f;
        instance.transform.values[2][2] = 1.0f;
        instance.transform.values[3][3] = 1.0f;
        const auto tlas = device.build_tlas(std::span<const gpu::Instance>(&instance, 1));
        if (!blas || !tlas)
            throw std::runtime_error("acceleration structure creation failed");

        const auto shader = device.create_shader(read_shader(GPU_RAYGEN_SHADER));
        const auto pipeline = device.ray_tracing({shader, {}, {}, {}, {}});
        auto output = device.buffer<std::uint32_t>(1);
        struct Args { gpu::GpuPtr<std::uint32_t> result_buffer; } args{output.ptr()};
        pipeline.trace({4, 3, 1}, args);
        device.synchronize();

        std::uint32_t actual = 0;
        device.download(std::span<std::uint32_t>(&actual, 1), output);
        if (actual != 23u)
            throw std::runtime_error("ray-generation result mismatch: got "
                + std::to_string(actual));
        std::cout << "ray-tracing example passed: BLAS/TLAS built, dispatch result="
                  << actual << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ray-tracing example failed: " << error.what() << '\n';
        return 1;
    }
}
