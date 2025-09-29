#pragma once
#include "MemoryPool.h"
#include <RmlUi/Core/Types.h>
#include <cstring>

class GeometryData : public GpuResource {
public:
    GeometryData(MemoryPool& pool, const Rml::Span<const Rml::Vertex> vertices, const Rml::Span<const int> indices) : GpuResource(GpuResourceType::Geometry), m_pool(pool), m_num_indices(indices.size())
    {
        // Allocate and copy
        const vk::DeviceSize vertex_buffer_size = vertices.size() * sizeof(Rml::Vertex);
        if (const auto alloc = m_pool.allocate(vertex_buffer_size)) {
            m_vertex_allocation = *alloc;
            memcpy(m_vertex_allocation.mapped_data, vertices.data(), vertex_buffer_size);
        }
        const vk::DeviceSize index_buffer_size = indices.size() * sizeof(int);
        if (const auto alloc = m_pool.allocate(index_buffer_size)) {
            m_index_allocation = *alloc;
            memcpy(m_index_allocation.mapped_data, indices.data(), index_buffer_size);
        }
    }

    ~GeometryData() override
    {
        m_pool.free(m_vertex_allocation);
        m_pool.free(m_index_allocation);
    }

    GeometryData(const GeometryData&) = delete;
    GeometryData& operator=(const GeometryData&) = delete;
    GeometryData(GeometryData&&) = delete;
    GeometryData& operator=(GeometryData&&) = delete;
    
    vk::DeviceSize getVertexOffset() const { return m_vertex_allocation.offset;  }
    vk::DeviceSize getIndexOffset() const { return m_index_allocation.offset;  }
    uint32_t getNumIndices() const { return m_num_indices; }

private:
    MemoryPool& m_pool;
    SubAllocation m_vertex_allocation;
    SubAllocation m_index_allocation;
    uint32_t m_num_indices = 0;
};