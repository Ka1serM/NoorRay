#pragma once

#include "Context.h"
#include "Buffer.h"

class Accel {
public:
    Accel(const Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type);

    Buffer buffer;
    vk::UniqueAccelerationStructureKHR accel;
    vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo;
};