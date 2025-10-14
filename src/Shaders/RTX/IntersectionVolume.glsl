#version 460

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : enable

#include "../SharedStructs.h"
#include "../Bindings.glsl"

void main() {
    // For debugging, assume the ray always hits at t = 1.0
    // Normally, you'd check if ray intersects the volume's bounds or density
    reportIntersectionEXT(1.0, 0); // 0 = primitiveIndex
}