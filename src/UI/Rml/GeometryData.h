#pragma once

#include <vulkan/vulkan.hpp>
#include "vk_mem_alloc.h"
#include <RmlUi/Core/Types.h>

#include "RmlUi/Core/Vertex.h"

/**
 * @brief Encapsulates a Vulkan buffer containing vertex and index data for a piece of RmlUi geometry.
 *
 * This class handles the creation, memory mapping, and destruction of the underlying Vulkan resources
 * using the Vulkan Memory Allocator (VMA). It is designed to be non-copyable but movable for
 * efficient ownership transfer.
 */
class GeometryData {
public:
    /**
     * @brief Constructs and uploads geometry to a GPU buffer.
     * @param allocator The VMA allocator instance.
     * @param vertices A span of vertices to upload.
     * @param indices A span of indices to upload.
     */
    GeometryData(VmaAllocator allocator, Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices);
    ~GeometryData();

    // Deleted copy semantics to prevent accidental resource duplication.
    GeometryData(const GeometryData&) = delete;
    GeometryData& operator=(const GeometryData&) = delete;

    // Enabled move semantics for efficient ownership transfer.
    GeometryData(GeometryData&& other) noexcept;
    GeometryData& operator=(GeometryData&& other) noexcept;

    // --- Accessors ---
    vk::Buffer getBuffer() const { return m_buffer; }
    vk::DeviceSize getVertexOffset() const { return m_vertex_offset; }
    vk::DeviceSize getIndexOffset() const { return m_index_offset; }
    int getNumIndices() const { return m_num_indices; }

private:
    VmaAllocator m_allocator = nullptr;

    vk::Buffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;

    vk::DeviceSize m_vertex_offset = 0;
    vk::DeviceSize m_index_offset = 0;
    int m_num_indices = 0;
};
