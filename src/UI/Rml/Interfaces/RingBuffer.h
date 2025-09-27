#pragma once

#include "vk_mem_alloc.h"

struct GeometryData : GpuResource {
    uint32_t vertex_offset_bytes;
    uint32_t index_offset_bytes;
    uint32_t num_indices;
};

// Helper struct to track which parts of the buffer are in use by which frame
struct FrameData {
    vk::Fence fence;
    size_t offset;
};

// Represents a sub-allocation within the large buffer
struct Allocation {
    uint8_t* mapped_data;
    size_t offset;
};

class RingBuffer {
public:
    RingBuffer(VmaAllocator allocator, vk::Device device, size_t size, vk::BufferUsageFlags usage);
    ~RingBuffer();

    // Rule of 5: Non-copyable, non-movable resource
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Call at the start of a frame to sync and get the new starting offset
    void beginFrame(vk::Fence frame_fence);

    // Allocate a chunk of memory from the buffer
    Allocation allocate(size_t size, size_t alignment = 256);

    vk::Buffer getBuffer() const { return m_buffer; }

private:
    VmaAllocator m_allocator;
    vk::Device m_device;

    VkBuffer m_buffer;
    VmaAllocation m_allocation;
    uint8_t* m_mapped_data;
    size_t m_size;

    size_t m_head = 0; // Current position for new allocations
    size_t m_tail = 0; // Position of the oldest in-flight frame

    // Tracks which fences are associated with which offsets in the ring
    std::deque<FrameData> m_in_flight_frames;
};

inline RingBuffer::RingBuffer(VmaAllocator allocator, vk::Device device, size_t size, vk::BufferUsageFlags usage)
    : m_allocator(allocator), m_device(device), m_size(size)
{
    auto buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo({}, m_size, usage));
    VmaAllocationCreateInfo alloc_ci = {VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU};
    vmaCreateBuffer(m_allocator, &buffer_ci, &alloc_ci, &m_buffer, &m_allocation, nullptr);
    vmaMapMemory(m_allocator, m_allocation, reinterpret_cast<void**>(&m_mapped_data));
}

inline RingBuffer::~RingBuffer()
{
    if (m_allocator) {
        vmaUnmapMemory(m_allocator, m_allocation);
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }
}

inline void RingBuffer::beginFrame(vk::Fence frame_fence)
{
    // Check fences for completed frames to reclaim space
    while (!m_in_flight_frames.empty()) {
        auto& oldest_frame = m_in_flight_frames.front();
        auto status = m_device.getFenceStatus(oldest_frame.fence);
        if (status == vk::Result::eSuccess) {
            m_tail = oldest_frame.offset;
            m_in_flight_frames.pop_front();
        } else
            break;
    }

    // Record the current position for the new frame
    m_in_flight_frames.emplace_back(frame_fence, m_head);
}

inline Allocation RingBuffer::allocate(size_t size, size_t alignment)
{
    // Align the head to the required alignment
    size_t aligned_head = (m_head + alignment - 1) & ~(alignment - 1);
    
    // Check if we need to wrap around to the beginning
    if (aligned_head + size > m_size)
        aligned_head = 0; // Wrap around

    bool is_full = false;
    // The buffer is only "full" in the traditional sense if head and tail are not at the same spot.
    // If head == tail, the buffer is empty.
    if (m_head != m_tail) {
        if (aligned_head >= m_head) { // Case 1: No wrap. Head is chasing tail.
            if (m_head < m_tail && aligned_head + size > m_tail) {
                is_full = true;
            }
        } else { // Case 2: We wrapped. Tail is chasing head.
            if (aligned_head + size > m_tail) {
                is_full = true;
            }
        }
    }
    
    // If the single allocation is larger than the entire buffer, it's also full.
    if (size > m_size)
        is_full = true;

    if (is_full)
        throw std::runtime_error("DynamicRingBuffer is full!");

    Allocation alloc;
    alloc.offset = aligned_head;
    alloc.mapped_data = m_mapped_data + aligned_head;

    // Advance the head
    m_head = aligned_head + size;

    return alloc;
}