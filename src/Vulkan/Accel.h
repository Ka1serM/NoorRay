#pragma once

#include "Context.h"
#include "Buffer.h"

class Accel {
public:
    Accel() = default;

    void build(Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type);

    //Rule of Five
    ~Accel() = default;
    Accel(const Accel&) = delete;
    Accel& operator=(const Accel&) = delete;
    Accel(Accel&&) noexcept = default;
    Accel& operator=(Accel&&) noexcept = default;
    
    Buffer buffer;
    vk::UniqueAccelerationStructureKHR accel;

    vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo{};
    vk::AccelerationStructureTypeKHR type{};
};