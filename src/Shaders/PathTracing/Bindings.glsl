#ifndef _BINDINGS_GLSL_ //include guard to prevent multiple inclusions
#define _BINDINGS_GLSL_

#ifdef USE_COMPUTE
    // For the Compute pipeline, binding 0 is the buffer of instance data.
layout(set = 0, binding = 0) buffer InstanceBuffer { ComputeInstance sceneInstances[]; };
#else
    // For the RTX pipeline, binding 0 is the Top-Level Acceleration Structure.
layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
#endif

// Binding 2: Mesh Data Pointers (vertex, index, material addresses, etc.)
layout(set = 0, binding = 2) buffer MeshAddressesBuffer { MeshAddresses instances[]; };

// Binding 3: Global Texture Sampler Array
layout(set = 0, binding = 3) uniform sampler2D textureSamplers[];


// --- Buffer Reference Type Definitions ---
layout(buffer_reference, scalar) buffer VertexBuffer { Vertex data[]; };
layout(buffer_reference, scalar) buffer IndexBuffer { uint data[]; };
layout(buffer_reference, scalar) buffer FaceBuffer { Face data[]; };
layout(buffer_reference, scalar) buffer MaterialBuffer { Material data[]; };
layout(buffer_reference, scalar) buffer BVHBuffer { BVHNode data[]; };

#endif // _BINDINGS_GLSL_