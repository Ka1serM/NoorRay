#pragma once

#include "Context.h"
#include "Buffer.h"
#include <vulkan/vulkan.hpp>

class Accel {
    Buffer buffer;
    vk::UniqueAccelerationStructureKHR accel;
    vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo{};
    vk::AccelerationStructureTypeKHR type{};

public:
    Accel() = default;
    Accel(Accel&& other) noexcept = default;
    Accel& operator=(Accel&& other) noexcept = default;

    void build(Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type);
    ~Accel() = default;
    
    // Getters
    const Buffer& getBuffer() const { return buffer; }
    const vk::AccelerationStructureKHR& getAccelerationStructure() const { return accel.get(); }
    const vk::WriteDescriptorSetAccelerationStructureKHR& getDescriptorAccelerationStructureInfo() const { return descAccelInfo; }
    vk::AccelerationStructureTypeKHR getType() const { return type; }
};