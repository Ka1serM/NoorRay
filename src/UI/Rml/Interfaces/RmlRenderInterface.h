#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <RmlUi/Core/Types.h>
#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include <map>
#include <vector>

#define RMLUI_VULKAN_ENABLE_CLIPPING 1
#define RMLUI_VULKAN_ENABLE_SHADERS 1

struct GpuResource {
    virtual ~GpuResource() = default;
};

#include "RingBuffer.h"

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
    #embed "../../../Shaders/Rml/RmlVert.spv"
};
static constexpr unsigned char shader_frag[] = {
    #embed "../../../Shaders/Rml/RmlFrag.spv"
};

#if RMLUI_VULKAN_ENABLE_SHADERS
static constexpr unsigned char shader_frag_gradient[] = {
    #embed "../../../Shaders/Rml/RmlGradientFrag.spv"
};
#endif

struct PushConstants {
    Rml::Matrix4f transform;   // 64 bytes
    Rml::Vector2f translate;   // 8 bytes
    int texture_id = -1;             // 4 bytes
};

#if RMLUI_VULKAN_ENABLE_SHADERS
#define MAX_STOPS 16
#define GRADIENT_LINEAR           0
#define GRADIENT_RADIAL           1
#define GRADIENT_CONIC            2

struct alignas(32) ColorStop {
    Rml::Colourf color;
    float position;
    // 12 bytes of padding are implicitly handled by the alignment
};

// `alignas(16)` ensures correct std140 alignment for members
struct alignas(16) GradientData {
    int gradient_function; int repeating;
    int num_stops; int _pad0; // Padding to align the following vec2
    Rml::Vector2f p;
    Rml::Vector2f v;
    ColorStop stops[MAX_STOPS];
};
#endif


class RmlRenderInterface : public Rml::RenderInterface
{
public:
    RmlRenderInterface(vk::Device device, vk::Queue graphics_queue, VmaAllocator allocator, vk::CommandPool command_pool,
        vk::DescriptorPool descriptor_pool, vk::Format colorFormat, vk::Format depthFormat);
    ~RmlRenderInterface() override;
    
    void beginFrame(vk::CommandBuffer command_buffer, vk::Image target_image, vk::ImageView target_image_view, vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fence);
    void endFrame();
    
    Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
    void RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

    Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions) override;
    Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) override;
    void ReleaseTexture(Rml::TextureHandle texture) override;
    
    void EnableScissorRegion(bool enable) override;
    void SetScissorRegion(Rml::Rectanglei region) override;
    void SetTransform(const Rml::Matrix4f* transform) override;

    void registerVulkanTexture(const Rml::String& name, vk::ImageView image_view, Rml::Vector2i dimensions);
    void unregisterVulkanTexture(const Rml::String& name);

#if RMLUI_VULKAN_ENABLE_CLIPPING
    void EnableClipMask(bool enable) override;
    void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) override;
#endif

#if RMLUI_VULKAN_ENABLE_SHADERS
    Rml::CompiledShaderHandle CompileShader(const Rml::String& name, const Rml::Dictionary& parameters) override;
    void RenderShader(Rml::CompiledShaderHandle shader, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    void ReleaseShader(Rml::CompiledShaderHandle shader) override;
#endif

private:
    void CreateDescriptors();
    void CreatePipelines(vk::Format colorFormat, vk::Format depthFormat);

    vk::Device m_device;
    vk::Queue m_queue;
    VmaAllocator m_allocator;
    vk::CommandPool m_command_pool;
    vk::DescriptorPool m_descriptor_pool;
    
    vk::UniqueDescriptorSetLayout m_textures_descriptor_set_layout;
    vk::UniqueDescriptorSet m_textures_descriptor_set;
    vk::UniqueSampler m_linear_sampler;
    vk::UniquePipelineLayout m_pipeline_textures_layout;
    vk::UniquePipeline m_pipeline_main;
    
#if RMLUI_VULKAN_ENABLE_SHADERS
    vk::UniqueDescriptorSetLayout m_ubo_descriptor_set_layout;
    vk::UniquePipelineLayout m_pipeline_ubo_textures_layout;
    vk::UniquePipeline m_pipeline_gradient;
#endif

#if RMLUI_VULKAN_ENABLE_CLIPPING
    vk::UniquePipeline m_pipeline_stencil_gen;
    vk::UniquePipeline m_pipeline_stencil_clip;
    vk::UniquePipeline m_pipeline_stencil_incr;
    vk::UniquePipeline m_pipeline_stencil_zero;
    bool m_clip_mask_enabled = false;
    uint32_t m_stencil_level = 0;
#endif

#if RMLUI_VULKAN_ENABLE_SHADERS && RMLUI_VULKAN_ENABLE_CLIPPING
    vk::UniquePipeline m_pipeline_gradient_stencil_clip;
#endif
    
    // Frame-specific State 
    vk::CommandBuffer m_command_buffer;
    vk::Fence m_current_frame_fence = nullptr;
    vk::ImageView m_target_image_view;
    vk::Image m_target_image;
    vk::Extent2D m_target_extent;
    Rml::Matrix4f m_projection_matrix;
    Rml::Matrix4f m_correction_matrix;
    Rml::Matrix4f m_transform_matrix;
    bool m_transform_enabled = false;
    bool m_scissor_enabled = true;
    vk::Rect2D m_current_scissor;

    // Resource Management 
    static constexpr uint32_t kMaxBindlessTextures = 1024;
    std::vector<uint32_t> m_free_texture_indices;
    std::map<Rml::String, std::pair<Rml::TextureHandle, Rml::Vector2i>> m_registered_textures;

    Rml::CompiledGeometryHandle m_fullscreen_quad;

    RingBuffer m_ring_buffer;
    std::deque<std::pair<vk::Fence, std::vector<GpuResource*>>> m_destruction_queue;
};