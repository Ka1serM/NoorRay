#ifndef _BINDINGS_GLSL_
#define _BINDINGS_GLSL_

#ifdef USE_COMPUTE
// For the Compute pipeline, binding 0 is the buffer of instance data
layout(set = 0, binding = 0) buffer InstanceBuffer { ComputeInstance instances[]; };
#else
// For the RTX pipeline, binding 0 is the Top-Level AS
layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
#endif

// G-buffer outputs
layout (set = 0, binding = 1, rgba32f) uniform image2D outputColor;
layout (set = 0, binding = 2, rgba8) uniform image2D outputAlbedo;
layout (set = 0, binding = 3, rgba16f) uniform image2D outputNormal;
layout (set = 0, binding = 4, r32ui) uniform uimage2D outputCrypto;
layout (set = 0, binding = 5, rgba16f) uniform image2D outputPosition;

// Asset data pointer
layout(set = 0, binding = 6) buffer AssetDataBuffer { AssetData assets[]; };

// Global texture array
layout(set = 0, binding = 7) uniform sampler2D textureSamplers[];

// Buffer-reference type definitions
layout(buffer_reference, scalar) buffer MeshBuffer   { MeshData data; };
layout(buffer_reference, scalar) buffer VolumeBuffer { VolumeData data; };

// Mesh data points
layout(buffer_reference, scalar) buffer VertexBuffer   { Vertex data[]; };
layout(buffer_reference, scalar) buffer IndexBuffer    { uint data[]; };
layout(buffer_reference, scalar) buffer FaceBuffer     { Face data[]; };
layout(buffer_reference, scalar) buffer MaterialBuffer { Material data[]; };

// Only compute RT
layout(buffer_reference, scalar) buffer BVHNodeBuffer  { BVHNode data[]; };
layout(buffer_reference, scalar) buffer BVHIndexBuffer { uint data[]; };

#endif