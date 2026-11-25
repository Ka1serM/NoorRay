#include <array>
#include <print>
#include <string_view>

#include "DiffRenderer.Tests/Tests.h"

namespace DiffRenderer::Tests {

Result check(const std::expected<void, NoorRHI::Error>& result) {
    if (!result)
        return std::unexpected(result.error().message);
    return {};
}

Result submit(NoorRHI::Device& device, NoorRHI::CommandList&& list) {
    std::expected<NoorRHI::Token, NoorRHI::Error> token = device.submit(std::move(list));
    if (!token)
        return std::unexpected(token.error().message);
    return check(device.wait(*token));
}

}

int main(const int count, char** arguments) {
    using namespace DiffRenderer::Tests;
    struct Test {
        std::string_view name;
        Result (*run)(NoorRHI::Device&);
    };
    constexpr std::array tests{
        Test{"loss", testLoss},
        Test{"scan", testScan},
        Test{"sort", testSort},
        Test{"adam", testAdam},
        Test{"comparison", testComparison},
        Test{"mesh-texture", testMeshTexture},
        Test{"sdf-method", testSdfMethod},
#ifdef NOORRAY_GAUSSIAN_TRACING
        Test{"gaussian-tracing", testGaussianTracing},
        Test{"gaussian-method", testGaussianMethod},
#endif
    };
    NoorRHI::Device device;
    if (std::expected<void, NoorRHI::Error> opened = device.open(); !opened) {
        std::println(stderr, "the GPU device does not open: {}", opened.error().message);
        return 1;
    }
    int failures = 0;
    for (const Test& test : tests) {
        if (count > 1 && std::string_view(arguments[1]) != test.name)
            continue;
        std::println("== {}", test.name);
        if (const Result result = test.run(device); !result) {
            std::println("FAILED: {}", result.error());
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
