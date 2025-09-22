#pragma once
#include <memory>
#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include "RmlUi/Core/Colour.h"
#include "RmlUi/Core/Math.h"
#include "RmlUi/Core/RenderInterface.h"
#include "RmlUi/Core/Types.h"
#include "RmlUi/Core/Vector2.h"

class UniformBuffer;
// Forward declare to avoid circular include
class RmlRenderInterface;

// --- Shaders for layer compositing and filters ---
static constexpr unsigned char shader_passthrough_frag[] = {
#embed "../../Shaders/Rml/RmlPassthroughFrag.spv"
};
static constexpr unsigned char shader_passthrough_vert[] = {
#embed "../../Shaders/Rml/RmlPassthroughVert.spv"
};
static constexpr unsigned char shader_frag_filter[] = {
    #embed "../../Shaders/Rml/RmlFilterFrag.spv"
};

// --- Filter-related structs and enums ---
enum class FilterType { Invalid, Opacity, Blur, DropShadow, ColorMatrix};

struct CompiledFilter {
    FilterType type = FilterType::Invalid;
    std::unique_ptr<UniformBuffer> ubo;
    float sigma = 0.0f;
    Rml::Vector2f drop_shadow_offset = {0.0f, 0.0f};
    Rml::ColourbPremultiplied drop_shadow_color;
};

#define FILTER_TYPE_OPACITY             0
#define FILTER_TYPE_COLOR_MATRIX        1
#define FILTER_TYPE_BLUR_VERTICAL       2
#define FILTER_TYPE_BLUR_HORIZONTAL     3
#define FILTER_TYPE_DROP_SHADOW_ALPHA   4
#define BLUR_NUM_WEIGHTS   4
        
struct FilterData {
    int filter_type;
    float scalar_value;
    Rml::Vector2f drop_shadow_offset;
    Rml::Colourf drop_shadow_color;
    float blur_weights[BLUR_NUM_WEIGHTS];
    float _blur_padding[4 - (BLUR_NUM_WEIGHTS % 4)];
    Rml::Matrix4f color_matrix;
};

/**
 * @brief Manages render layers, post-processing buffers, and filter effects.
 * This class encapsulates all offscreen rendering, compositing, and filter application logic,
 * separating it from the main RmlRenderInterface.
 */
class RenderLayerManager {
public:
    // Represents an offscreen render target (image, view, descriptor set).
    class RenderLayer {
    public:
        RenderLayer(VmaAllocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format,
                    vk::DescriptorPool descriptor_pool, vk::DescriptorSetLayout texture_layout, vk::Sampler sampler);
        ~RenderLayer() = default;

        // Movable but not copyable
        RenderLayer(const RenderLayer&) = delete;
        RenderLayer& operator=(const RenderLayer&) = delete;
        RenderLayer(RenderLayer&& other) noexcept = default;
        RenderLayer& operator=(RenderLayer&& other) noexcept = default;

        vk::Image image;
        vk::UniqueImageView image_view;
        VmaAllocation allocation;
        vk::DescriptorSet texture_descriptor_set = VK_NULL_HANDLE;
        vk::Extent2D extent;
    };

public:
    RenderLayerManager(
        VmaAllocator allocator,
        vk::Device device,
        vk::DescriptorPool descriptor_pool,
        vk::DescriptorSetLayout texture_layout,
        vk::DescriptorSetLayout ubo_layout,
        vk::Sampler sampler,
        Rml::CompiledGeometryHandle fullscreen_quad,
        vk::Format color_format,
        vk::Format depth_format,
        vk::Extent2D initial_extent
    );
    ~RenderLayerManager();

    // Not copyable or movable
    RenderLayerManager(const RenderLayerManager&) = delete;
    RenderLayerManager& operator=(const RenderLayerManager&) = delete;
    RenderLayerManager(RenderLayerManager&&) = delete;
    RenderLayerManager& operator=(RenderLayerManager&&) = delete;

    // --- Frame Lifecycle ---
    void beginFrame(vk::CommandBuffer command_buffer, vk::Fence in_flight_fence, vk::Extent2D target_extent);
    void endFrame(vk::ImageView target_image_view);
    
    // --- Layer Management ---
    Rml::LayerHandle PushLayer();
    void PopLayer();
    void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters);

    // --- Filter Management ---
    Rml::CompiledFilterHandle CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters);
    void ReleaseFilter(Rml::CompiledFilterHandle filter);
    
    // --- State Update ---
    void SetClipMaskEnabled(bool enabled) { m_clip_mask_enabled = enabled; }

private:
    void CreatePipelines(vk::Format color_format, vk::Format depth_format);
    void CreatePostProcessResources(vk::Extent2D extent);

    // --- Filter Rendering ---
    void RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filters);
    void RenderSingleFilterPass(const CompiledFilter* filter);
    static void CalculateBlurWeights(float sigma, FilterData& out_data);

    // --- Core Vulkan Objects ---
    VmaAllocator m_allocator;
    vk::Device m_device;
    vk::DescriptorPool m_descriptor_pool;
    vk::DescriptorSetLayout m_texture_layout;
    vk::DescriptorSetLayout m_ubo_layout;
    vk::Sampler m_sampler;
    
    // --- Pipelines and Shaders ---
    vk::UniqueShaderModule m_vert_passthrough;
    vk::UniqueShaderModule m_frag_passthrough;
    vk::UniqueShaderModule m_frag_shader_filter;
    vk::UniquePipelineLayout m_passthrough_layout;
    vk::UniquePipelineLayout m_filter_layout;
    vk::UniquePipeline m_passthrough_pipeline;
    vk::UniquePipeline m_passthrough_replace_pipeline;
    vk::UniquePipeline m_filter_pipeline;
    vk::UniquePipeline m_filter_stencil_clip_pipeline;

    // --- Frame-specific State ---
    vk::CommandBuffer m_command_buffer;
    vk::Fence m_current_frame_fence = nullptr;
    vk::Extent2D m_target_extent;
    bool m_clip_mask_enabled = false;
    
    // --- Managed Resources ---
    std::vector<RenderLayer> m_layers;
    std::array<std::unique_ptr<RenderLayer>, 2> m_postprocess_buffers;
    int m_postprocess_idx = 0;
    Rml::CompiledGeometryHandle m_fullscreen_quad;
    std::map<vk::Fence, std::vector<RenderLayer>> m_destruction_map_layers;
};