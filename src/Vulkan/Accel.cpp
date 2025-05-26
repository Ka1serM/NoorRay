#include "Accel.h"

#include <memory>

Accel::Accel(const std::shared_ptr<Context> &context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type) {
    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
    buildGeometryInfo.setType(type);
    buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
    buildGeometryInfo.setGeometries(geometry);

    // Create buffer
    vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context->device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount
    );

    vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
    buffer = Buffer(context, Buffer::Type::AccelStorage, size);

    // Create accel
    vk::AccelerationStructureCreateInfoKHR accelInfo;
    accelInfo.setBuffer(buffer.getBuffer());
    accelInfo.setSize(size);
    accelInfo.setType(type);
    accel = context->device->createAccelerationStructureKHRUnique(accelInfo);

    // Build
    Buffer scratchBuffer{context, Buffer::Type::Scratch, buildSizesInfo.buildScratchSize};
    buildGeometryInfo.setScratchData(scratchBuffer.getDeviceAddress());
    buildGeometryInfo.setDstAccelerationStructure(*accel);

    context->oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {
        //
        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo;
        buildRangeInfo.setPrimitiveCount(primitiveCount);
        buildRangeInfo.setFirstVertex(0);
        buildRangeInfo.setPrimitiveOffset(0);
        buildRangeInfo.setTransformOffset(0);
        commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
    });

    descAccelInfo.setAccelerationStructures(*accel);
}
