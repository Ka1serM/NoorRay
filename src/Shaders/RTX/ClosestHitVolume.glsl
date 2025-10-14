#version 460
#extension GL_EXT_ray_tracing          : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference     : require
#extension GL_EXT_scalar_block_layout  : enable

#include "../SharedStructs.h"
#include "../Bindings.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;
hitAttributeEXT vec3 attribs;

void main()
{
    // Debug: solid red hit response
    payload.attenuation = vec3(1.0, 0.0, 0.0); // red
    payload.emission    = vec3(0.0);           // no emission
    payload.albedo      = vec3(1.0, 0.0, 0.0); // red albedo
    payload.roughness   = 0.5;
    payload.normal      = vec3(0.0, 1.0, 0.0); // up
    payload.objectIndex = uint(gl_InstanceID);
    payload.position    = gl_WorldRayOriginEXT + gl_RayTmaxEXT * gl_WorldRayDirectionEXT;
}