#include <noorrhi/noorrhi.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
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

}

TEST_CASE("NoorRHI API supplies the device used by NoorRay") {
    std::unique_ptr<noorrhi::Device> device;
    try {
        device = std::make_unique<noorrhi::Device>();
    } catch (const std::exception&) {
        SKIP("GPU device requires extensions unavailable on this Vulkan device");
    }

    auto shader = device->create_shader(read_shader(NOORRHI_TEST_SHADER));
    auto pipeline = device->compute(shader);
    auto input = device->buffer<float>(1);
    auto output = device->buffer<float>(1);
    const float value = 41.0f;
    input.upload(std::span<const float>(&value, 1));
    struct Args {
        noorrhi::GpuPtr<float> input;
        noorrhi::GpuPtr<float> b;
        noorrhi::GpuPtr<float> output;
        std::uint32_t count;
    } args{input.ptr(), input.ptr(), output.ptr(), 1};
    pipeline.launch({1, 1, 1}, args);
    device->synchronize();

    float actual = 0;
    output.download(std::span<float>(&actual, 1));
    REQUIRE(actual == 82.0f);
}
