#include "Accel.h"

void Accel::build(Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type) {
    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
    buildGeometryInfo.setType(type);
    buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
    buildGeometryInfo.setGeometries(geometry);

    // Get sizes required for AS buffers
    vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context.getDevice().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount
    );

    // Allocate buffer for acceleration structure storage
    vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
    buffer = Buffer(context, Buffer::Type::AccelStorage, size);

    // Create acceleration structure
    vk::AccelerationStructureCreateInfoKHR accelInfo{};
    accelInfo.setBuffer(buffer.getBuffer());
    accelInfo.setSize(size);
    accelInfo.setType(type);
    accel = context.getDevice().createAccelerationStructureKHRUnique(accelInfo);

    // Create scratch buffer
    Buffer scratchBuffer{context, Buffer::Type::Scratch, buildSizesInfo.buildScratchSize};
    buildGeometryInfo.setScratchData(scratchBuffer.getDeviceAddress());
    buildGeometryInfo.setDstAccelerationStructure(*accel);

    // Build range info
    vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.setPrimitiveCount(primitiveCount);
    buildRangeInfo.setFirstVertex(0);
    buildRangeInfo.setPrimitiveOffset(0);
    buildRangeInfo.setTransformOffset(0);

    // Submit build command once
    context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {
        commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
    });

    // Update descriptor info for binding
    descAccelInfo.setAccelerationStructures(*accel);
}