// Traces one ray per launch index at a triangle in the z=0 plane and records
// whether it hit, so the test exercises the acceleration structure, the shader
// binding table, and the miss/closest-hit groups rather than just the raygen.

// Unbounded resource arrays are the descriptor-heap idiom: the index is the
// opaque handle the library returns for the acceleration structure.
RaytracingAccelerationStructure accelerationStructures[];

struct Arguments
{
    uint* result;
    uint width;
    uint scene;
};

struct Root
{
    Arguments* arguments;
};

[[vk::push_constant]] ConstantBuffer<Root> root;

struct Payload
{
    uint hit;
};

[shader("raygeneration")]
void rayGenMain()
{
    const Arguments arguments = *root.arguments;
    const uint2 launchId = DispatchRaysIndex().xy;
    const uint2 launchSize = DispatchRaysDimensions().xy;

    // Map the launch grid onto [-1, 1] in the triangle's plane and shoot
    // straight down -z, so the centre hits and the corners miss.
    const float2 uv = (float2(launchId) + 0.5f) / float2(launchSize) * 2.0f - 1.0f;

    RayDesc ray;
    ray.Origin = float3(uv, 1.0f);
    ray.Direction = float3(0.0f, 0.0f, -1.0f);
    ray.TMin = 0.001f;
    ray.TMax = 10.0f;

    Payload payload;
    payload.hit = 0u;
    TraceRay(accelerationStructures[arguments.scene], RAY_FLAG_NONE, 0xff,
        0, 0, 0, ray, payload);

    arguments.result[launchId.y * arguments.width + launchId.x] = payload.hit;
}

[shader("miss")]
void missMain(inout Payload payload)
{
    payload.hit = 0u;
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    payload.hit = 1u;
}
