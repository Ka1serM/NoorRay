#ifndef _BINDINGS_SLANG_
#define _BINDINGS_SLANG_

[[vk::binding(0, 0)]] StructuredBuffer<Instance, Std430DataLayout> instances;
[[vk::binding(1, 0)]] [format("rgba32f")] RWTexture2D<float4> outputColor;
[[vk::binding(2, 0)]] [format("rgba8")] RWTexture2D<float4> outputAlbedo;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> outputNormal;
[[vk::binding(4, 0)]] [format("r32ui")] RWTexture2D<uint> outputCrypto;
[[vk::binding(5, 0)]] [format("rgba16f")] RWTexture2D<float4> outputPosition;
[[vk::binding(6, 0)]] [format("rgba16f")] RWTexture2D<float4> outputMaterial;
#define outputAdaptiveState outputMaterial
[[vk::binding(7, 0)]] StructuredBuffer<MeshAddresses, Std430DataLayout> meshes;
[[vk::binding(8, 0)]] StructuredBuffer<SceneSettings, ScalarDataLayout> sceneSettingsBuffer;
#define sceneSettings sceneSettingsBuffer[0]
[[vk::binding(9, 0)]] StructuredBuffer<LightGpu, Std430DataLayout> sceneLightsBuffer;
[[vk::binding(10, 0)]] StructuredBuffer<RayLutEntry, Std430DataLayout> cameraRayLut;
[[vk::binding(11, 0)]] Sampler2D textureSamplers[];

// Pointer wrappers
struct VertexBuffer { Vertex data[]; };
struct IndexBuffer { uint data[]; };
struct FaceBuffer { Face data[]; };
struct MaterialBuffer { Material data[]; };
struct BVHNodeBuffer { BVHNode data[]; };
struct BVHIndexBuffer { uint data[]; };

#endif
