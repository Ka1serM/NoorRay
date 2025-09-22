#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <RmlUi/Core/Types.h>
#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include <map>
#include <vector>
#include <memory>

#include "GeometryData.h"

class TextureData;
class RenderLayerManager;

// TGA Header struct for texture loading
#pragma pack(1)
struct TGAHeader {
    char idLength;
    char colourMapType;
    char dataType;
    short int colourMapOrigin;
    short int colourMapLength;
    char colourMapDepth;
    short int xOrigin;
    short int yOrigin;
    short int width;
    short int height;
    char bitsPerPixel;
    char imageDescriptor;
};
#pragma pack()

static constexpr unsigned char shader_vert[] = {
#embed "../../Shaders/Rml/RmlVert.spv"
};
static constexpr unsigned char shader_frag[] = {
#embed "../../Shaders/Rml/RmlFrag.spv"
};
static constexpr unsigned char shader_frag_gradient[] = {
    #embed "../../Shaders/Rml/RmlGradientFrag.spv"
};

struct PushConstants {
    Rml::Matrix4f transform;
    Rml::Vector2f translate;
    int texture_id = -1;
};

struct GradientData {
    int gradient_function; // 0: linear, 1: radial, 2: conic, +3 for repeating
    int num_stops;
    float _padding1, _padding2;
    Rml::Vector2f p, v;
    struct {
        float position;
        float _padding[3];
        Rml::Colourf color;
    } stops[16];
};


class RmlRenderInterface : public Rml::RenderInterface
{
public:
    RmlRenderInterface(
        vk::Device device,
        vk::Queue graphics_queue,
        VmaAllocator allocator,
        vk::CommandPool command_pool,
        vk::DescriptorPool descriptor_pool,
        vk::Format colorFormat,
        vk::Format depthFormat,
        vk::Extent2D initialExtent
    );
    ~RmlRenderInterface() override;
    
    // --- Frame Lifecycle ---
    void beginFrame(vk::CommandBuffer command_buffer, vk::Image target_image, vk::ImageView target_image_view, vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fence);
    void endFrame();
    
    // --- Core Rendering ---
    Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
    void RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

    // --- Textures ---
    Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions) override;
    Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) override;
    void ReleaseTexture(Rml::TextureHandle texture) override;
    void registerVulkanTexture(const Rml::String& name, vk::ImageView image_view, Rml::Vector2i dimensions);
    void unregisterVulkanTexture(const Rml::String& name);
    
    // --- State Management ---
    void EnableScissorRegion(bool enable) override;
    void SetScissorRegion(Rml::Rectanglei region) override;
    void SetTransform(const Rml::Matrix4f* transform) override;
    void EnableClipMask(bool enable) override;
    void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) override;

    // --- Shaders (Gradients) ---
    Rml::CompiledShaderHandle CompileShader(const Rml::String& name, const Rml::Dictionary& parameters) override;
    void RenderShader(Rml::CompiledShaderHandle shader, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    void ReleaseShader(Rml::CompiledShaderHandle shader) override;

#if RMLUI_VULKAN_ENABLE_LAYERS
    // --- Layers & Filters (Now delegated to RenderLayerManager) ---
    Rml::LayerHandle PushLayer() override;
    void PopLayer() override;
    void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters) override;
    Rml::CompiledFilterHandle CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters) override;
    void ReleaseFilter(Rml::CompiledFilterHandle filter) override;
#endif

private:
    void CreateDescriptors();
    void CreatePipelines(vk::Format colorFormat, vk::Format depthFormat);

    // --- Core Vulkan Objects ---
    vk::Device m_device;
    vk::Queue m_graphics_queue;
    VmaAllocator m_allocator;
    vk::CommandPool m_utility_command_pool;
    vk::DescriptorPool m_descriptor_pool;
    
    // --- Descriptors ---
    vk::UniqueDescriptorSetLayout m_textures_descriptor_set_layout;
    vk::UniqueDescriptorSetLayout m_ubo_descriptor_set_layout;
    vk::UniqueDescriptorSetLayout m_texture_descriptor_set_layout;
    vk::UniqueDescriptorSet m_textures_descriptor_set;
    vk::UniqueSampler m_linear_sampler;
    
    // --- Pipelines ---
    vk::UniquePipelineLayout m_pipeline_textures_layout;
    vk::UniquePipelineLayout m_pipeline_ubo_textures_layout;
    vk::UniquePipeline m_pipeline_main;
    vk::UniquePipeline m_pipeline_stencil_gen;
    vk::UniquePipeline m_pipeline_stencil_clip;
    vk::UniquePipeline m_pipeline_stencil_incr;
    vk::UniquePipeline m_pipeline_stencil_zero;
    vk::UniquePipeline m_pipeline_gradient;
    vk::UniquePipeline m_pipeline_gradient_stencil_clip;
    
    // --- Frame-specific State ---
    vk::CommandBuffer m_command_buffer;
    vk::Fence m_current_frame_fence = nullptr;
    vk::ImageView m_target_image_view;
    vk::Image m_target_image;
    vk::Extent2D m_target_extent;
    Rml::Matrix4f m_projection_matrix;
    Rml::Matrix4f m_transform_matrix;
    bool m_transform_enabled = false;
    bool m_scissor_enabled = true;
    bool m_clip_mask_enabled = false;
    uint32_t m_stencil_level = 0;
    vk::Rect2D m_current_scissor;
    
    // --- Resource Management ---
    static constexpr uint32_t kMaxBindlessTextures = 1024;
    std::vector<uint32_t> m_free_texture_indices;
    Rml::CompiledGeometryHandle m_fullscreen_quad = {};
    std::map<Rml::String, std::pair<Rml::TextureHandle, Rml::Vector2i>> m_registered_textures;

    // --- Deferred Destruction Queues ---
    std::map<vk::Fence, std::vector<std::unique_ptr<GeometryData>>> m_destruction_map_geometry;
    std::map<vk::Fence, std::vector<std::unique_ptr<TextureData>>> m_destruction_map_textures;
    
    std::unique_ptr<RenderLayerManager> m_render_layer_manager;
};
