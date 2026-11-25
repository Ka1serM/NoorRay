import NoorRHI;

#include "NoorRay/Shaders/Common/GpuRecords.h"

//! The context of a camera ray with no camera: Camera Data and the window coordinates read 0 until setCamera, and the object has no colour or pass index.
void setCameraRay(inout ShadingContext context) {
    context.worldToCamera0 = float4(0.0, 0.0, 0.0, 0.0);
    context.worldToCamera1 = float4(0.0, 0.0, 0.0, 0.0);
    context.worldToCamera2 = float4(0.0, 0.0, 0.0, 0.0);
    context.cameraWindow = float4(0.0, 0.0, 0.0, 0.0);
    context.rayType = float4(1.0, 0.0, 0.0, 0.0);
    context.rayEvent = float4(0.0, 0.0, 0.0, 0.0);
    context.rayDepth = float4(0.0, 0.0, 0.0, 0.0);
    context.objectColor = float4(1.0, 1.0, 1.0, 1.0);
    context.objectInfo = float4(0.0, 0.0, 1.0, 0.0);
}

//! The camera rows turn NoorRay camera space, which looks along -z, into Cycles camera space, which looks along +z.
void setCamera(inout ShadingContext context, CameraRecord camera) {
    context.worldToCamera0 = camera.worldToCamera[0];
    context.worldToCamera1 = camera.worldToCamera[1];
    context.worldToCamera2 = -camera.worldToCamera[2];
    const bool perspective = camera.meta.x != cameraTypeOrthographic;
    const float2 sensor = max(camera.sensor.xy, float2(1e-9, 1e-9));
    const float2 scale = perspective ? camera.projection.x / sensor : float2(camera.sensor.y / (camera.projection.y * sensor.x), 1.0 / camera.projection.y);
    context.cameraWindow = float4(scale, 0.5 - camera.sensor.zw / sensor);
    context.objectInfo.z = perspective ? 1.0 : 0.0;
}
