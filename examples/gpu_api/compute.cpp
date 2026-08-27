#include <gpu/gpu.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <span>
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
        constexpr std::size_t count = 257;
        gpu::Device device;
        const auto shader = device.create_shader(read_shader(GPU_COMPUTE_SHADER));
        auto lhs = device.buffer<float>(count);
        auto rhs = device.buffer<float>(count);
        auto output = device.buffer<float>(count);

        std::array<float, count> left{};
        std::array<float, count> right{};
        for (std::size_t i = 0; i < count; ++i) {
            left[i] = static_cast<float>(i);
            right[i] = 100.0f - static_cast<float>(i) * 0.5f;
        }
        device.upload(lhs, std::span<const float>(left));
        device.upload(rhs, std::span<const float>(right));

        struct Args {
            gpu::GpuPtr<float> lhs;
            gpu::GpuPtr<float> rhs;
            gpu::GpuPtr<float> result;
            std::uint32_t count;
        } args{lhs.ptr(), rhs.ptr(), output.ptr(), static_cast<std::uint32_t>(count)};

        device.compute(shader).launch({(args.count + 63u) / 64u, 1, 1}, args);
        device.barrier(gpu::Stage::Compute, gpu::Stage::Copy);
        device.synchronize();

        std::array<float, count> actual{};
        device.download(std::span<float>(actual), output);
        for (std::size_t i = 0; i < count; ++i) {
            const float expected = left[i] + right[i];
            if (actual[i] != expected)
                throw std::runtime_error("compute result mismatch at element "
                    + std::to_string(i) + ": got " + std::to_string(actual[i])
                    + ", expected " + std::to_string(expected));
        }
        std::cout << "compute example passed: " << count
                  << " additions, sample=" << actual[123] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "compute example failed: " << error.what() << '\n';
        return 1;
    }
}
