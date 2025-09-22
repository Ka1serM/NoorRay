#include "GeometryData.h"

#include "RmlUi/Core/Vertex.h"

GeometryData::GeometryData(VmaAllocator allocator, Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices)
    : m_allocator(allocator), m_num_indices(static_cast<int>(indices.size()))
{
    const vk::DeviceSize vertices_size = vertices.size() * sizeof(Rml::Vertex);
    const vk::DeviceSize indices_size = indices.size() * sizeof(int);
    const vk::DeviceSize total_size = vertices_size + indices_size;

    m_vertex_offset = 0;
    m_index_offset = vertices_size;

    // Create a single buffer for both vertices and indices.
    const auto buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo(
        {}, 
        total_size, 
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer
    ));

    // Allocate memory that is host-visible and mappable for direct CPU-to-GPU transfer.
    constexpr VmaAllocationCreateInfo alloc_ci = {
        VMA_ALLOCATION_CREATE_MAPPED_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU
    };
    
    VmaAllocationInfo alloc_info = {};
    VkBuffer raw_buffer;
    vmaCreateBuffer(m_allocator, &buffer_ci, &alloc_ci, &raw_buffer, &m_allocation, &alloc_info);
    m_buffer = raw_buffer;

    // Copy data into the mapped buffer.
    memcpy(alloc_info.pMappedData, vertices.data(), vertices_size);
    memcpy(static_cast<std::byte*>(alloc_info.pMappedData) + vertices_size, indices.data(), indices_size);
}

GeometryData::~GeometryData() {
    if (m_buffer && m_allocation) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }
}

GeometryData::GeometryData(GeometryData&& other) noexcept
    : m_allocator(other.m_allocator), m_buffer(other.m_buffer), m_allocation(other.m_allocation),
      m_vertex_offset(other.m_vertex_offset), m_index_offset(other.m_index_offset), m_num_indices(other.m_num_indices)
{
    // Invalidate the other object to prevent double-free.
    other.m_buffer = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
}

GeometryData& GeometryData::operator=(GeometryData&& other) noexcept {
    if (this != &other) {
        // Destroy existing resources before overwriting.
        this->~GeometryData();

        // Move new resources.
        m_allocator = other.m_allocator;
        m_buffer = other.m_buffer;
        m_allocation = other.m_allocation;
        m_vertex_offset = other.m_vertex_offset;
        m_index_offset = other.m_index_offset;
        m_num_indices = other.m_num_indices;

        // Invalidate the other object.
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
    }
    return *this;
}