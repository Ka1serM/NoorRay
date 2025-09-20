#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <RmlUi/Core/Types.h>
#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include <map>

#pragma pack(1)
struct TGAHeader {
    char idLength, colourMapType, dataType;
    short int colourMapOrigin, colourMapLength;
    char colourMapDepth;
    short int xOrigin, yOrigin, width, height;
    char bitsPerPixel, imageDescriptor;
};
#pragma pack()

static constexpr unsigned char shader_frag[] = {
    #embed "../../Shaders/Rml/RmlFrag.spv"
};
static constexpr unsigned char shader_vert[] = {
    #embed "../../Shaders/Rml/RmlVert.spv"
};
static constexpr unsigned char shader_frag_gradient[] = {
#embed "../../Shaders/Rml/RmlGradientFrag.spv"
};


class RmlRenderInterface : public Rml::RenderInterface {
public:
    static constexpr uint32_t kMaxBindlessTextures = 1024;

    explicit RmlRenderInterface(
       vk::Device device,
       vk::Queue graphics_queue,
       VmaAllocator allocator,
       vk::CommandPool command_pool,
       vk::DescriptorPool descriptor_pool,
       vk::Format colorFormat,
       vk::Format depthFormat
    );
    ~RmlRenderInterface() override;
    
    // --- Rml::RenderInterface Overrides ---
    Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
    void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

    Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) override;
    Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source_data, Rml::Vector2i source_dimensions) override;
    void ReleaseTexture(Rml::TextureHandle texture_handle) override;

    void EnableScissorRegion(bool enable) override;
    void SetScissorRegion(Rml::Rectanglei region) override;
    void SetTransform(const Rml::Matrix4f* transform) override;

    void EnableClipMask(bool enable) override;
    void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) override;
    
    Rml::CompiledShaderHandle CompileShader(const Rml::String& name, const Rml::Dictionary& parameters) override;
    void RenderShader(Rml::CompiledShaderHandle shader, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    void ReleaseShader(Rml::CompiledShaderHandle shader) override;

    /**
 * @brief Registers an existing, application-owned Vulkan ImageView under a unique name for RmlUi.
 * RmlUi can then load this texture using the path "vulkan://<name>".
 * The application remains the owner of the ImageView and is responsible for its lifetime.
 * @param name The unique identifier for the texture (e.g., "my_render_target").
 * @param image_view The Vulkan ImageView handle for the texture. This handle is NOT owned by the interface.
 * @param dimensions The width and height of the texture.
 */
    void registerVulkanTexture(const Rml::String& name, vk::ImageView image_view, Rml::Vector2i dimensions);

    /**
     * @brief Unregisters a texture previously registered. This removes the name lookup
     * and queues the associated RmlUi-internal resources for safe release. It does NOT
     * affect the application-owned ImageView.
     * @param name The unique identifier used during registration.
     */
    void unregisterVulkanTexture(const Rml::String& name);
    
    // --- Frame Rendering ---
    void beginFrame(vk::CommandBuffer command_buffer, vk::ImageView target_image_view, vk::ImageView depthImageView, vk::Extent2D target_extent, vk::Fence in_flight_fence);
    void endFrame() const;

private:

    #define RMLUI_MAX_COLOR_STOPS 16
    enum class ShaderGradientFunction { Linear, Radial, Conic, RepeatingLinear, RepeatingRadial, RepeatingConic };
    
    struct GradientUBO {
        GradientUBO(VmaAllocator allocator, vk::Device device) : m_allocator(allocator), m_device(device) {}
        ~GradientUBO() {
            if (buffer && allocation)
                vmaDestroyBuffer(m_allocator, buffer, allocation);
        }

        VmaAllocator m_allocator;
        vk::Device m_device;
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        vk::UniqueDescriptorSet descriptor_set;
    };

    // Defines the type of a compiled shader.
    enum class CompiledShaderType {
        Invalid,
        Gradient,
    };

    // Holds all data for a compiled shader, including its UBO if it has one.
    struct CompiledShader {
        CompiledShaderType type = CompiledShaderType::Invalid;
        std::unique_ptr<GradientUBO> gradient_ubo;
    };

    // Mirrors the ColorStop struct in the gradient fragment shader.
    // `alignas(32)` ensures correct std140 alignment for the struct array.
    struct alignas(32) ColorStop {
        Rml::Colourf color;
        float position;
        // 12 bytes of padding are implicitly handled by the alignment
    };

    // Mirrors the GradientUBO uniform block in the gradient fragment shader.
    // `alignas(16)` ensures correct std140 alignment for its members.
    struct alignas(16) GradientData {
        int gradient_function;
        int num_stops;
        float _pad[2]; // Padding to align the following vec2
        Rml::Vector2f p;
        Rml::Vector2f v;
        ColorStop stops[16];
    };

    struct PushConstants {
        Rml::Matrix4f transform;   // 64 bytes
        Rml::Vector2f translate;   // 8 bytes
        int texture_id;             // 4 bytes
        int padding0;               // pad to 16 bytes
    };
    
    struct TextureData {
        // Filled only if we create the image/view ourselves (from file or data).
        vk::Image image = nullptr;
        VmaAllocation allocation = nullptr;
        vk::UniqueImageView owned_image_view;
        // A sampler for this texture. May be the shared linear sampler or a unique one.
        vk::Sampler sampler = nullptr;
        // This is used for textures registered via `registerVulkanTexture`.
        // The interface will NEVER destroy this.
        vk::ImageView image_view_raw = nullptr; 
        uint32_t bindless_index = static_cast<uint32_t>(-1);
        // Helper to get the correct, non-owned ImageView handle for descriptor updates.
        vk::ImageView getImageView() const { return owned_image_view ? owned_image_view.get() : image_view_raw; }
    };

    struct GeometryData {
       vk::Buffer buffer = nullptr;
       VmaAllocation allocation = nullptr;
       int num_indices;
       vk::DeviceSize vertex_offset;
       vk::DeviceSize index_offset;
    };
    
    // Helper to create a TextureData wrapper for an existing, non-owned ImageView
    Rml::TextureHandle CreateTextureHandleForView(vk::ImageView image_view);

    vk::Device m_device;
    vk::Queue m_graphics_queue;
    VmaAllocator m_allocator;
    vk::CommandPool m_utility_command_pool;
    vk::DescriptorPool m_descriptor_pool;

    vk::CommandBuffer m_command_buffer;
    vk::Rect2D m_current_scissor;

    vk::UniqueDescriptorSetLayout m_bindless_descriptor_set_layout;
    vk::DescriptorSet m_bindless_descriptor_set;
    vk::UniquePipelineLayout m_pipeline_layout;

    vk::UniqueDescriptorSetLayout m_gradient_descriptor_set_layout;
    
    vk::UniquePipeline m_pipeline_main;
    
    vk::UniquePipeline m_pipeline_gradient;
    vk::UniquePipeline m_pipeline_gradient_stencil_clip;
    vk::UniquePipelineLayout m_pipeline_gradient_layout;
    
    vk::UniquePipeline m_pipeline_stencil_gen;
    vk::UniquePipeline m_pipeline_stencil_clip;
    vk::UniquePipeline m_pipeline_stencil_incr;
    vk::UniquePipeline m_pipeline_stencil_zero;
    
    vk::UniqueSampler m_linear_sampler;

    Rml::Matrix4f m_projection_matrix;
    Rml::Matrix4f m_transform_matrix;
    bool m_transform_enabled = false;
    bool m_scissor_enabled = false;
    bool m_clip_mask_enabled = false;
    int m_stencil_level = 0;

    // Garbage collection for GPU resources, deferred by frame fences
    std::map<vk::Fence, std::vector<std::unique_ptr<GeometryData>>> m_destruction_map_geometry;
    std::map<vk::Fence, std::vector<std::unique_ptr<TextureData>>> m_destruction_map_textures;
    vk::Fence m_current_frame_fence = nullptr;
    
    Rml::Vector<uint32_t> m_free_texture_indices;
    
    // Map for textures registered by the user application.
    // Maps a URI name to the RmlUi texture handle and its dimensions.
    std::map<Rml::String, std::pair<Rml::TextureHandle, Rml::Vector2i>> m_registered_textures;
    
    void CreateDescriptors();
    void CreatePipelines(vk::Format colorFormat, vk::Format depthFormat);
    void DestroyTexture(const TextureData* texture);
};