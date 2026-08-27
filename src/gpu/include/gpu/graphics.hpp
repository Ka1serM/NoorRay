#pragma once

#include "image.hpp"
#include "shader.hpp"

#include <functional>
#include <memory>
#include <type_traits>

namespace gpu {

enum class CullMode { None, Front, Back, FrontAndBack };
enum class FrontFace { CounterClockwise, Clockwise };
enum class PolygonMode { Fill, Line, Point };
enum class CompareOp { Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always };

struct BlendState {
    bool enabled = false;
};
struct GraphicsState {
    CullMode cull = CullMode::Back;
    FrontFace front_face = FrontFace::CounterClockwise;
    PolygonMode polygon = PolygonMode::Fill;
    CompareOp depth_compare = CompareOp::LessOrEqual;
    bool depth_test = false;
    bool depth_write = false;
    BlendState blend{};
};
struct GraphicsPipelineDesc {
    Shader vertex;
    Shader fragment;
    GraphicsState state{};
    ImageFormat color_format = ImageFormat::Rgba8Unorm;
};
struct RenderTarget {
    ImageHandle color{};
    ImageHandle depth{};
    // Clear the attachments on entry. Setting this false preserves what is
    // already in the target, which is what compositing a later pass on top of
    // an earlier one needs.
    bool clear = true;
    // Flip the viewport to the Y-up NDC convention glm's projection matrices
    // assume, instead of Vulkan's native Y-down.
    bool flip_y = false;
};

namespace detail { class GraphicsPipelineImpl; }

class GraphicsPipeline {
public:
    GraphicsPipeline() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

    void draw(std::uint32_t vertex_count) const;
    void draw_instanced(std::uint32_t vertex_count, std::uint32_t instance_count) const;
    void draw_indirect(GpuPtr<DrawArgs> commands, std::uint32_t draw_count = 1) const;
    template<class Args>
    void draw(std::uint32_t vertex_count, const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        draw_bytes(vertex_count, &args, sizeof(Args));
    }
    template<class Args>
    void draw_instanced(std::uint32_t vertex_count, std::uint32_t instance_count,
        const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        draw_instanced_bytes(vertex_count, instance_count, &args, sizeof(Args));
    }
    // The draw commands are read from GPU memory; the root arguments are
    // supplied by the host exactly as they are for a direct draw.
    template<class Args>
    void draw_indirect(GpuPtr<DrawArgs> commands, std::uint32_t draw_count,
        const Args& args) const {
        static_assert(std::is_trivially_copyable_v<Args>, "GPU arguments must be trivially copyable");
        draw_indirect_bytes(commands, draw_count, &args, sizeof(Args));
    }

private:
    friend class Device;
    explicit GraphicsPipeline(std::shared_ptr<detail::GraphicsPipelineImpl> impl)
        : impl_(std::move(impl)) {}
    void draw_bytes(std::uint32_t vertex_count, const void* args, std::size_t size) const;
    void draw_instanced_bytes(std::uint32_t vertex_count, std::uint32_t instance_count,
        const void* args, std::size_t size) const;
    void draw_indirect_bytes(GpuPtr<DrawArgs> commands, std::uint32_t draw_count,
        const void* args, std::size_t size) const;
    std::shared_ptr<detail::GraphicsPipelineImpl> impl_;
};

} // namespace gpu
