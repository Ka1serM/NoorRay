#include <gpu/gpu.hpp>

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
        gpu::Device device;
        const auto shader_bytes = read_shader(GPU_GRAPHICS_SHADER);
        const auto vertex = device.create_shader(shader_bytes, "vertMain");
        const auto fragment = device.create_shader(shader_bytes, "fragMain");
        const auto pipeline = device.graphics({vertex, fragment, {},
            gpu::ImageFormat::Rgba8Unorm});
        const auto target = device.image<std::uint8_t>(128, 128,
            gpu::ImageUsage::ColorAttachment, gpu::ImageFormat::Rgba8Unorm);

        device.render({target.handle(), {}}, [&] { pipeline.draw(3); });
        device.synchronize();

        // Read the result back: the centre must carry the triangle, and a
        // corner must still hold the clear colour.
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(target.width()) * target.height() * 4u);
        device.download(std::span<std::uint8_t>(pixels), target);
        const std::size_t centre =
            (static_cast<std::size_t>(target.height() / 2) * target.width()
                + target.width() / 2) * 4u;
        const bool centre_lit = pixels[centre] || pixels[centre + 1] || pixels[centre + 2];
        const bool corner_clear = !pixels[0] && !pixels[1] && !pixels[2];
        if (!centre_lit || !corner_clear)
            throw std::runtime_error("rendered image did not contain the triangle");
        std::cout << "graphics example passed: rendered a triangle to "
                  << target.width() << 'x' << target.height() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graphics example failed: " << error.what() << '\n';
        return 1;
    }
}
