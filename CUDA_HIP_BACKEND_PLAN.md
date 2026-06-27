# NoorRay: CUDA 13.3 + OptiX 9.1 Wavefront Backend Plan

## Coding Style Requirements

All new code must match the existing NoorRay codebase conventions. Key rules:

| Aspect | Convention | Example |
|--------|-----------|---------|
| Include guard | `#pragma once` | `#pragma once` |
| Class names | PascalCase | `GpuWavefrontRaytracer`, `WavefrontQueues` |
| Method names | camelCase | `getFrameInfo()`, `render()`, `setup()` |
| Member variables | No prefix, plain camelCase | `vkCtx`, `scene`, `queues`, `nextBuffer` |
| Struct members | Same as members | `origin`, `direction`, `sampleIndex` |
| Padding fields | Leading underscore | `_pad0`, `_pad1` |
| Enum classes | PascalCase type + values, explicit base | `enum class GpuBackend : int { None, CUDA };` |
| Macros | SCREAMING_SNAKE_CASE | `NR_CUDA_ACTIVE`, `NR_GPU_CHECK()` |
| Constants | `static constexpr` or `#define` | `static constexpr int MAX_TEXTURES = 256;` |
| Namespaces | None (global scope) | — |
| Indentation | 4 spaces, no tabs | |
| Brace style | Allman (brace on its own line) for class/struct | `class Foo\n{\n` |
| Include order | stdlib `<...>` → project `"..."` → third-party `"..."` | `<cstdint>` before `"GPU/Annotations.h"` |
| Struct alignment | Explicit `alignas()` for GPU ABI structs | `struct alignas(16) PathState {` |

This applies to all new files (`src/GPU/*`, `src/Kernels/*`, `src/Raytracing/GpuWavefrontRaytracer.*`).

## Decisions Summary

| Question | Decision |
|----------|----------|
| Initial platform | Linux + NVIDIA CUDA only. No HIP source files, targets, dependencies, or stubs are added now; unused backend `#else` branches remain empty. |
| GPU memory | Device-local allocations for hot queues, path state, geometry, and counters. Managed memory is restricted to cold setup data and must be prefetched before rendering. |
| Texture ownership | Pure CUDA — `cudaArray_t` + `cudaTextureObject_t`; no Vulkan involvement for textures |
| VkAccelerationStructure | **Removed entirely** — Vulkan is display/UI only, no ray queries |
| Binary model | One CUDA-enabled NoorRay executable; startup reports a clear error if the required NVIDIA CUDA/OptiX device is unavailable. |
| Toolchain | Installed CUDA 13.3 and OptiX 9.1 headers from `/home/marcel/Programs/OptixSDK` |
| GPU target | RTX 4070, Ada Lovelace, `sm_89`; regular CUDA kernels compile for architecture `89`, OptiX device code for `compute_89`. |
| Display interop | Zero-copy from day one: Vulkan allocates AOV `VkImage`s with export → CUDA writes via mapped arrays; semaphore sync |
| Math types | GLM everywhere — `glm::vec3`/`glm::vec4` in host and CUDA device code; define `GLM_FORCE_CUDA` for CUDA translation units. Default non-aligned `glm::vec3` is 12 B/alignment 4. |
| Raytracer hierarchy | `GpuWavefrontRaytracer` is the only renderer constructed or called. Existing Vulkan renderer sources remain in the repository as unbuilt reference code. |
| Render model | Async launch on main thread — `render()` fires kernels onto a CUDA stream and returns immediately (non-blocking). Vulkan waits on the GPU via semaphore at v-sync. CPU never blocks. No background threads. |
| Queue model | Two compacted ping-pong ray queues. Hit and shadow work use the active-ray index directly; only primary/continuation ray compaction uses warp-aggregated atomics. No pop atomics, per-material queues, or `Advance` pass. |
| Feature scope | Full behavioral parity: every current camera including realistic-camera LUT, samplers, adaptive sampling, materials, lights, accumulation, six AOVs, scene updates, and UI controls. |
| OptiX packaging | Compile OptiX programs to OptiX IR with `nvcc --optix-ir --gpu-architecture=compute_89`; convert IR to a generated C++ byte array with CUDA `bin2c` and embed it in the executable. |
| Interop synchronization | One imported Vulkan/CUDA timeline semaphore with monotonically increasing values. CUDA waits for buffer release before reuse; Vulkan waits for render completion before sampling it. |

---

## Overview

Replace the active Vulkan compute/RTX path tracing pipeline with native CUDA and OptiX, keeping Vulkan for the swapchain, ImGui, tonemapper, and final display blit. The old Vulkan renderer remains in the repository for reference but is excluded from the executable and never instantiated.

Shared kernel headers compile under ordinary C++ and `nvcc`. Annotation macros expand to CUDA attributes only under `__CUDACC__` and to nothing for host-only compilation. Hot rendering data stays in device-local memory. Host-visible configuration is uploaded asynchronously; no queue count is read back while rendering.

### Performance and Race-Safety Invariants

- Consumers never pop work atomically. A thread consumes item `idx` only when `idx < rayCounts[depth]`.
- `Extend` writes `hitQueue[idx]`; misses use an invalid primitive sentinel. No hit counter or hit append atomic exists.
- `Shade` writes `shadowQueue[idx]`; paths without a shadow ray write an invalid sentinel. No shadow counter or shadow append atomic exists.
- Only ray queues are compacted. `Generate` compacts pixels selected by adaptive sampling into depth 0, and `Shade` compacts surviving continuation rays. A warp reserves one contiguous range with one atomic add, then its active lanes write that range.
- Ray counts are indexed by path depth and zeroed once at frame start. No counter is reset while producers may still access it.
- Ray queues ping-pong by depth: input is `rayQueues[depth & 1]`, output is `rayQueues[(depth + 1) & 1]`. They are distinct allocations.
- One active ray and at most one shadow ray exist per sample at a depth. `Shade` and `Connect` therefore never concurrently update the same `PathState`; the stream order between passes is sufficient.
- No per-material queues. A single general `Shade` pass evaluates the material selected by the hit.
- No CPU synchronization or device-to-host queue-count transfer occurs between passes.
- OptiX launch parameters are immutable per pass/depth while work is in flight. Extend and Connect each own a device-local parameter array indexed by depth; no asynchronous launch observes a host-overwritten parameter block.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  NoorRay Process                                                      │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Vulkan  (display + UI only)                                  │   │
│  │                                                               │   │
│  │  Swapchain → Present                                          │   │
│  │  ImGui render pass                                            │   │
│  │  Tonemapper (frag shader, reads shared VkImage AOVs)          │   │
│  │  VkSemaphore for CUDA↔Vulkan sync                             │   │
│  └───────────────────────┬───────────────────────────────────────┘   │
│                           │ shared VkImage (external memory)         │
│  ┌────────────────────────▼──────────────────────────────────────┐   │
│  │  GpuWavefrontRaytracer  (host-side driver)                    │   │
│  │                                                               │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │  CUDA 13.3 / OptiX 9.1 backend (RTX 4070, sm_89)      │   │   │
│  │  │  optixLaunch (Extend/Connect), CUDA kernels (others)  │   │   │
│  │  │  cudaMallocAsync device-local hot storage             │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  GPU Memory Layout                                                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Wavefront queues│  │ Mesh vertex/idx  │  │ AOV images       │   │
│  │ device-local    │  │ device-local     │  │ VkImage(export)  │   │
│  │ (CUDA only)     │  │ (CUDA only)      │  │ ← CUDA writes    │   │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘   │
│                         ┌──────────────────┐                         │
│                         │ Textures         │                         │
│                         │ cudaArray_t      │                         │
│                         │ (CUDA only)      │                         │
│                         └──────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
src/
├── GPU/                                  # Portability layer — header-only except Memory.cpp
│   ├── Annotations.h                     # NR_CPU_GPU / NR_GPU / NR_GPU_KERNEL / NR_GPU_DEVICE_COMPILE
│   ├── Checks.h                          # NR_GPU_CHECK(call) — host throws, device __trap()
│   ├── Memory.h / Memory.cpp             # Device-local allocator; managed allocator for cold data only
│   ├── Allocator.h                       # Backend-aware GPU allocators
│   ├── RayTraversal.h                    # OptiX device traversal helpers
│   ├── ImageInterop.h / .cpp             # Import Vulkan AOV images + timeline semaphore into CUDA
│   └── CudaDevice.h / .cpp               # CUDA device selection matched to Vulkan UUID
│
├── Kernels/                              # Shared host/CUDA headers and CUDA entry points
│   ├── Types.h                           # PathState, work items, WavefrontQueues — NR_CPU_GPU structs
│   ├── Queues.h                          # Direct-index access + warp-aggregated continuation append
│   ├── Samplers.h                        # R2, Halton — ported from Slang
│   ├── Geometry.h                        # Triangle intersection, barycentric interp
│   ├── Cameras.h                         # All camera models (Perspective/ThinLens/Fisheye/Realistic)
│   ├── Materials.h                       # PBR BXDF
│   ├── Lights.h                          # Light sampling, NEE helpers
│   │
│   ├── passes/                           # Kernel bodies — NR_GPU functions, no __global__ here
│   │   ├── Generate.h
│   │   ├── Extend.h                      # Closest-hit traversal body
│   │   ├── Shade.h
│   │   ├── Connect.h                     # Shadow ray traversal
│   │   └── Finalize.h
│   │
│   ├── cuda/                             # CUDA/OptiX compilation units — compiled by nvcc
│   │   ├── Kernels.cu                    # __global__ launchers for non-RT passes
│   │   ├── OptixPrograms.cu              # Embedded-IR Extend and Connect raygen programs
│   │   ├── AccelBuilder.cu               # optixAccelBuild BLAS + TLAS
│   │   ├── OptiXSetup.cu                 # OptiX context, embedded IR modules, pipeline, SBT
│   │   └── generated/                    # Build-tree bin2c OptiX IR headers; never edited
│
├── Raytracing/
│   ├── GpuWavefrontRaytracer.h / .cpp    # Sole active renderer
│   ├── Raytracer.h / .cpp                # Retained reference, excluded from build
│   ├── WavefrontRaytracer.h / .cpp       # Retained reference, excluded from build
│   ├── ComputeRaytracer.h / .cpp         # Retained reference, excluded from build
│   └── RtxRaytracer.h / .cpp             # Retained reference, excluded from build
│
└── Vulkan/                               # Unchanged — display + UI only; AS code removed
    ├── Context.h / .cpp
    ├── Buffer.h / .cpp
    └── ...
```

---

## Phase 1: CUDA Foundation

### Annotations

```cpp
// src/GPU/Annotations.h
#pragma once

#if defined(NR_BACKEND_CUDA)
  #define NR_CUDA_ACTIVE 1
#endif

// ── Device-side compile guard ─────────────────────────────────────
#if defined(__CUDA_ARCH__)
  #define NR_GPU_DEVICE_COMPILE 1
#endif

// ── Decoration macros ─────────────────────────────────────────────
#if defined(__CUDACC__)
  #define NR_CPU_GPU    __host__ __device__
  #define NR_GPU        __device__
  #define NR_GPU_KERNEL __global__
#else
  #define NR_CPU_GPU
  #define NR_GPU
  #define NR_GPU_KERNEL
#endif

// ── Launch index abstraction ──────────────────────────────────────
// These are referenced only from CUDA device functions.
#if defined(NR_GPU_DEVICE_COMPILE)
  #define NR_GPU_LAUNCH_IDX       (blockIdx.x * blockDim.x + threadIdx.x)
  #define NR_GPU_OPTIX_LAUNCH_ID  optixGetLaunchIndex().x
#endif
```

### Error Checking

```cpp
// src/GPU/Checks.h
#pragma once
#include "Annotations.h"

#if defined(NR_CUDA_ACTIVE)
  #include <cuda_runtime.h>
  #define NR_GPU_CHECK(call)  do { \
      cudaError_t _e = (call); \
      if (_e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(_e)); \
  } while(0)
  #define NR_OPTIX_CHECK(call) do { \
      OptixResult _r = (call); \
      if (_r != OPTIX_SUCCESS) throw std::runtime_error(optixGetErrorName(_r)); \
  } while(0)
#else
  #define NR_GPU_CHECK(call)  (call)
  #define NR_OPTIX_CHECK(call) (call)
#endif
```

### GPU Memory

All renderer-owned GPU allocations use `cudaMallocAsync` and `cudaFreeAsync` on the render stream with a long-lived CUDA memory pool. Queue storage, path state, counters, vertices, indices, materials, and texture-handle arrays are device-local and never dereferenced by the CPU during rendering. This avoids managed-memory page faults and keeps frame latency predictable.

```cpp
// src/GPU/Memory.h
#pragma once
#include <cstddef>
namespace nr::gpu {
    void* malloc_device(std::size_t bytes, cudaStream_t stream);
    void  free_device(void* p, cudaStream_t stream) noexcept;
}
```

Initial uploads use asynchronous host-to-device copies from reusable pinned staging memory on the render stream. Rendering never reads counters or path state back to the host. The renderer synchronizes only during setup, resize, scene rebuild, and destruction; steady-state `render()` remains asynchronous.

The only steady-state queue atomic is `atomicAdd` inside the warp-aggregated primary/continuation ray reservation.

---

## Phase 2: Shared Kernel Types

**File:** `src/Kernels/Types.h`

Port the Slang structs to plain C++ using host/device-compatible GLM types. This file becomes the active CUDA source of truth. The existing `Shared.h` remains only because the unbuilt Vulkan reference renderer still includes it.

```cpp
// src/Kernels/Types.h
#pragma once
#include "GPU/Annotations.h"
#include <cstdint>

// GLM used in host and CUDA device code.
// GLM_FORCE_CUDA must be defined before the first GLM include
// in CUDA translation units to mark types __host__ __device__.
// Do not alias these as float3/float4: CUDA owns those global names.
// Default glm::vec3 remains layout-compatible: size 12, alignment 4.
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat4 = glm::mat4;

// Hot state read and written by Shade/Connect/Finalize.
struct alignas(16) PathState {
    Vec3     throughput;
    Vec3     radiance;
    uint32_t rngState;
    uint32_t depth;
    uint32_t flags;
    uint32_t packedCounters;
    uint32_t lastBsdfPdf_bits;   // bit-cast float
    uint32_t _pad0;
};
static_assert(sizeof(PathState) == 48);

// Cold first-hit AOV state kept out of the hot path-state cache lines.
struct alignas(16) PrimaryState {
    Vec3     primaryAlbedo;
    Vec3     primaryNormal;
    Vec3     primaryPosition;
    uint32_t primaryObjectIndex;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(PrimaryState) == 48);

struct alignas(16) PathRayWorkItem {
    Vec3     origin;
    uint32_t sampleIndex;
    Vec3     direction;
    uint32_t _pad0;
};
static_assert(sizeof(PathRayWorkItem) == 32);

struct alignas(16) HitWorkItem {
    Vec3     rayDirection;
    uint32_t sampleIndex;
    float    baryU, baryV;
    uint32_t instanceIndex;
    uint32_t primitiveIndex;
};
static_assert(sizeof(HitWorkItem) == 32);

struct alignas(16) ShadowWorkItem {
    Vec3     origin;
    float    tMin;
    Vec3     direction;
    float    tMax;
    Vec3     contribution;
    uint32_t sampleIndex;
};
static_assert(sizeof(ShadowWorkItem) == 48);

// rayCounts has maxBounces + 1 entries and is cleared once per sample/frame.
struct WavefrontQueues {
    uint32_t*         rayCounts;
    PathState*        pathStates;
    PrimaryState*     primaryStates;
    PathRayWorkItem*  rayQueues[2];
    HitWorkItem*      hitQueue;
    ShadowWorkItem*   shadowQueue;
    uint32_t          capacity;
};
```

`HitWorkItem::primitiveIndex == ~0u` marks a miss. `ShadowWorkItem::tMax <= tMin` marks an unused shadow slot. These one-to-one queues have `capacity` entries and require no append counters. All pointed-to storage is device-local.

---

## Phase 3: Ray Traversal Abstraction

**File:** `src/GPU/RayTraversal.h`

All kernel code calls `intersect_ray()`; the correct API resolves at compile time via `#ifdef`.

```cpp
// src/GPU/RayTraversal.h
#pragma once
#include "Annotations.h"
#include "Kernels/Types.h"

struct RayHit {
    float    t             = 1e30f;
    float    u             = 0.f, v = 0.f;
    uint32_t instanceIndex = ~0u;
    uint32_t primIndex     = ~0u;
    bool     hit           = false;
};

#include <optix.h>
using SceneTraversable = OptixTraversableHandle;

// ── Inline intersection ───────────────────────────────────────────
NR_GPU inline RayHit intersect_ray(
    SceneTraversable accel,
    Vec3 origin, Vec3 direction,
    float tMin, float tMax)
{
    RayHit hit{};
#if defined(NR_GPU_DEVICE_COMPILE)
    // OptiX 9.1 Hit Object API, called from an OptiX raygen program.
    optixTraverse(
        accel, make_float3(origin.x, origin.y, origin.z),
        make_float3(direction.x, direction.y, direction.z), tMin, tMax,
        0.f,                           // rayTime
        0xFF,                          // visibilityMask
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0                        // SBT offsets
    );
    if (optixHitObjectIsHit()) {
        hit.hit           = true;
        hit.t             = optixHitObjectGetRayTmax();
        hit.u             = __uint_as_float(optixHitObjectGetAttribute_0());
        hit.v             = __uint_as_float(optixHitObjectGetAttribute_1());
        hit.instanceIndex = optixHitObjectGetInstanceIndex();
        hit.primIndex     = optixHitObjectGetPrimitiveIndex();
    }
#endif
    return hit;
}

NR_GPU inline bool test_occlusion(
    SceneTraversable accel,
    Vec3 origin, Vec3 direction,
    float tMin, float tMax)
{
#if defined(NR_GPU_DEVICE_COMPILE)
    optixTraverse(accel, make_float3(origin.x, origin.y, origin.z),
                  make_float3(direction.x, direction.y, direction.z), tMin, tMax,
                  0.f, 0xFF, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 1, 0);
    return optixHitObjectIsHit();
#else
    return false;
#endif
}
```

---

## Phase 4: Wavefront Kernel Passes

**Files:** `src/Kernels/passes/*.h`

Each pass exposes an `NR_GPU` function with the full kernel body. No `__global__` declarations live in the pass headers; entry points live in CUDA translation units.

All pass functions receive the thread index as a parameter. The entry point extracts it using `NR_GPU_LAUNCH_IDX` (regular kernel) or `NR_GPU_OPTIX_LAUNCH_ID` (OptiX raygen) and passes it through.

```cpp
// src/Kernels/passes/Extend.h
#pragma once
#include "GPU/Annotations.h"
#include "GPU/RayTraversal.h"
#include "Kernels/Types.h"
#include "Kernels/Queues.h"

NR_GPU inline void extend_pass(
    SceneTraversable accel, WavefrontQueues q, uint32_t depth, uint32_t idx)
{
    const uint32_t activeCount = q.rayCounts[depth];
    if (idx >= activeCount) return;

    const PathRayWorkItem ray = q.rayQueues[depth & 1u][idx];
    RayHit hit = intersect_ray(accel, ray.origin, ray.direction, 1e-4f, 1e10f);

    HitWorkItem item{};
    item.rayDirection = ray.direction;
    item.sampleIndex = ray.sampleIndex;
    if (hit.hit) {
        item.baryU         = hit.u;
        item.baryV         = hit.v;
        item.instanceIndex = hit.instanceIndex;
        item.primitiveIndex = hit.primIndex;
    } else {
        item.primitiveIndex = ~0u;
    }
    q.hitQueue[idx] = item;
}
```

`Generate` evaluates adaptive-sampling eligibility and warp-compacts active primary rays into `rayQueues[0]`. `Shade` consumes `hitQueue[idx]` for `idx < rayCounts[depth]`. It handles the miss sentinel, evaluates the general material path for hits, and always writes `shadowQueue[idx]` with either valid shadow work or the invalid sentinel. Surviving continuation rays are appended to `rayQueues[(depth + 1) & 1]` and increment `rayCounts[depth + 1]` through the same warp-aggregated reservation.

`Connect` consumes `shadowQueue[idx]` for the same active count, skips invalid entries, and calls `test_occlusion()` for valid entries. Because each active path owns exactly one slot and passes execute in stream order, no path-state write races exist.

There is no `Advance` pass. All depth counters are cleared once before `Generate`; queue selection and counter selection derive from `depth`.

### CUDA Entry Points

```cpp
// src/Kernels/cuda/Kernels.cu
// Non-RT passes: regular __global__ kernels
#include "Kernels/passes/Generate.h"
#include "Kernels/passes/Shade.h"
#include "Kernels/passes/Finalize.h"

__global__ void generate_kernel(SceneSettings s, WavefrontQueues q, uint32_t sampleIdx)
{ uint32_t idx = NR_GPU_LAUNCH_IDX; generate_pass(s, q, sampleIdx, idx); }

__global__ void shade_kernel(SceneData d, WavefrontQueues q)
{ uint32_t idx = NR_GPU_LAUNCH_IDX; shade_pass(d, q, idx); }

__global__ void finalize_kernel(WavefrontQueues q, OutputSurfaces output, uint32_t sampleIdx)
{ uint32_t idx = NR_GPU_LAUNCH_IDX; finalize_pass(q, outputImage, sampleIdx, idx); }
```

```cpp
// src/Kernels/cuda/OptixPrograms.cu
// Extend + Connect are two raygen entry points in one embedded OptiX module.
#include "Kernels/passes/Extend.h"
#include "Kernels/passes/Connect.h"

extern "C" __global__ void __raygen__extend()   {
    uint32_t idx = NR_GPU_OPTIX_LAUNCH_ID;
    extend_pass(optixGetSbtDataPointer_accel(), ..., idx);
}
extern "C" __global__ void __raygen__connect()  {
    uint32_t idx = NR_GPU_OPTIX_LAUNCH_ID;
    connect_pass(..., idx);
}
```

---

## Phase 5: OptiX Pipeline Setup

**Files:** `src/Kernels/cuda/OptiXSetup.cu` + `AccelBuilder.cu`

### Context and Pipeline

```
OptiXSetup.cu:
  optixInit()
  optixDeviceContextCreate(cudaContext, ...) → OptixDeviceContext

  // One module loaded from the generated embedded OptiX IR byte array
  optixModuleCreate(ctx, embeddedOptixIr, compileOptions) → module

  // Two raygen groups plus one triangle hitgroup with no CH/AH program.
  // optixTraverse returns a hit object directly and invokes neither miss nor closest-hit.
  optixProgramGroupCreate({ __raygen__extend }) → extGroup
  optixProgramGroupCreate({ __raygen__connect }) → connGroup
  optixProgramGroupCreate({ built-in triangle intersection }) → triangleHitGroup

  // Pipeline
  OptixPipelineCompileOptions:
    numPayloadValues = 0
    traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
    usesMotionBlur = false
  optixPipelineCreate(ctx, extGroup + connGroup + triangleHitGroup) → pipeline

  // SBT — one raygen record per pass and one hitgroup record per mesh/material range
  Fill raygen records with pass-specific data (queue pointers, TLAS handle)

  // Launch-state lifetime
  Allocate persistent device params extendParams[maxBounces]
  Allocate persistent device params connectParams[maxBounces]
  Build separate persistent Extend and Connect OptixShaderBindingTable structs
  Upload/update params only during synchronized setup, resize, or scene rebuild
```

### Acceleration Structure Build

```
AccelBuilder.cu:
  // Per-mesh BLAS
  OptixBuildInput triangleInput:
    vertexFormat   = OPTIX_VERTEX_FORMAT_FLOAT3
    vertexStrideBytes = sizeof(Vertex)
    vertexBuffers  = [device-local vertex buffer pointer]
    indexFormat    = OPTIX_INDICES_FORMAT_UNSIGNED_INT3
    indexBuffer    = device-local index buffer pointer

  optixAccelComputeMemoryUsage() → sizes
  scratchBuf = cudaMalloc(sizes.tempSizeInBytes)
  blasBuf    = cudaMalloc(sizes.outputSizeInBytes)
  optixAccelBuild(ctx, stream, buildInput, blasBuf, scratchBuf) → blasHandle

  // TLAS with instances
  OptixInstance[] instances — one per MeshInstance, contains:
    transform (float[12])
    instanceId
    blasHandle (from above)
  optixAccelBuild(ctx, stream, instanceInput) → tlasHandle

  // Rebuild policy:
  //   Mesh changed → rebuild BLAS + TLAS
  //   Transform changed → refit TLAS only (OPTIX_BUILD_FLAG_ALLOW_UPDATE)
```

---

## Phase 6: Vulkan External Memory Interop

All shared resources originate from Vulkan (with `VK_KHR_external_memory`) and are imported into CUDA via opaque FD handles. One `cudaExternalSemaphore_t` per frame synchronises the two queues.

### AOV Output Images (CUDA writes → Vulkan reads)

Six images are double-buffered (two complete sets × 6 = 12 VkImages). Their existing formats are preserved exactly:

| AOV | Vulkan format | CUDA channel/surface type |
|-----|---------------|---------------------------|
| Color | `VK_FORMAT_R32G32B32A32_SFLOAT` | 4 x 32-bit float / `float4` |
| Albedo | `VK_FORMAT_R8G8B8A8_UNORM` | 4 x 8-bit unsigned / `uchar4` |
| Normal | `VK_FORMAT_R16G16B16A16_SFLOAT` | 4 x 16-bit float / `half4` |
| Cryptomatte | `VK_FORMAT_R32_UINT` | 1 x 32-bit unsigned / `uint32_t` |
| Position | `VK_FORMAT_R16G16B16A16_SFLOAT` | 4 x 16-bit float / `half4` |
| Material | `VK_FORMAT_R16G16B16A16_SFLOAT` | 4 x 16-bit float / `half4` |

Each Vulkan image uses dedicated exportable memory. The CUDA external-memory mipmapped-array descriptor must match that image's exact channel format; it is not universally `float4`.

```
// Allocation (once, on resize) — repeated for buf ∈ {0, 1}:
VkExternalMemoryImageCreateInfo extInfo{};
extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
// attach to VkImageCreateInfo for each AOV

vkAllocateMemory (with VkExportMemoryAllocateInfo) → VkDeviceMemory

// Export to CUDA
vkGetMemoryFdKHR(device, {VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, mem}) → fd

cudaExternalMemoryHandleDesc memDesc{};
memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
memDesc.handle.fd = fd;
memDesc.size = imageMemoryRequirements.size;
cudaImportExternalMemory(&extMem[buf], &memDesc);

cudaExternalMemoryMipmappedArrayDesc arrDesc{};
arrDesc.formatDesc = matchingCudaChannelDesc(aovFormat);
arrDesc.extent     = make_cudaExtent(width, height, 0);
arrDesc.numLevels  = 1;
cudaExternalMemoryGetMappedMipmappedArray(&extMipArray, extMem[buf], &arrDesc);

cudaGetMipmappedArrayLevel(&cudaArray, extMipArray, 0);
cudaCreateSurfaceObject(&aovSurfaces[buf][i], &cudaArray);
```

The raytracer writes to `aovSurfaces[m_nextBuffer][]`; Vulkan reads `aovSurfaces[m_lastLaunched][]`.

### Textures (pure CUDA, no Vulkan)

Textures have no Vulkan involvement. On load they go straight into CUDA texture memory:

```cpp
// On texture load:
cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>(); // or float4
cudaMallocArray(&cudaArray, &fmt, width, height);
cudaMemcpy2DToArray(cudaArray, 0, 0, pixels, width*4, width*4, height, cudaMemcpyHostToDevice);

cudaResourceDesc resDesc{};
resDesc.resType         = cudaResourceTypeArray;
resDesc.res.array.array = cudaArray;

cudaTextureDesc texDesc{};
texDesc.filterMode      = cudaFilterModeLinear;
texDesc.addressMode[0]  = cudaAddressModeWrap;
texDesc.addressMode[1]  = cudaAddressModeWrap;
texDesc.normalizedCoords = 1;

cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

// Upload texObj values into a device-local array indexed by material texId.
// Kernel samples: tex2D<float4>(textureObjects[texId], u, v)
```

ImGui material thumbnails that need to display textures get a separate small CPU-side pixel read or a dedicated low-res VkImage upload — the path tracing textures are CUDA-only.

### Timeline Semaphore Sync (frame-precise GPU handoff)

Two AOV buffer sets ping-pong. Two exported timeline semaphores provide a bidirectional ownership handshake without serializing independent buffers:

- `renderReady`: CUDA signals frame value `F`; Vulkan waits for `F` before sampling that frame's AOV buffer.
- `bufferReleased`: Vulkan signals `F`; CUDA waits for the last frame value that used a buffer before writing that buffer again.

Both are Vulkan timeline semaphores exported as opaque FDs and imported into CUDA as `cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd`. CUDA takes ownership of each exported FD after a successful import.

```
VkSemaphore renderReady + bufferReleased
  → VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
  → cudaImportExternalSemaphore

Main thread — render():
  uint32_t buf = nextBuffer;              // 0 or 1
  uint64_t frameValue = ++submittedFrame;

  if (lastUseValue[buf] != 0)
      cudaWaitExternalSemaphoresAsync(bufferReleased, lastUseValue[buf], stream);

  // Fire all path tracing kernels onto stream (non-blocking)
  clearRayCounts();                       // one async clear per sample
  launchKernel(generate, buf);
  for (bounce..) { launchExtend(bounce); shade(bounce); launchConnect(bounce); }
  launchKernel(finalize, buf);

  cudaSignalExternalSemaphoresAsync(renderReady, frameValue, stream);

  nextBuffer = 1 - buf;                   // flip for next frame
  lastUseValue[buf] = frameValue;
  lastLaunched = {buf, frameValue};
  // → returns immediately, CPU never stalls

Main thread — Vulkan submit:
  vkQueueSubmit(wait timeline renderReady >= frameValue, ...)
  → Tonemapper reads AOV set [lastLaunched], ImGui, present
  → signal timeline bufferReleased = frameValue after the tonemapper is done
```

This prevents both hazards: Vulkan cannot sample unfinished CUDA output, and CUDA cannot overwrite a buffer still being sampled by Vulkan. All waits and signals are queued on GPU timelines; the CPU does not poll or block.

---

## Phase 7: Host-Side Driver

`GpuWavefrontRaytracer` is a standalone class and the only renderer instantiated by NoorRay. Existing Vulkan renderer files remain unchanged where practical as reference material, but CMake excludes their `.cpp` files and their constructors are not called.

**New file:** `src/Raytracing/GpuWavefrontRaytracer.h/.cpp`

No background threads. `render()` launches all kernel work onto a CUDA stream and returns immediately. The Vulkan v-sync callback submits the tonemapper with a semaphore wait — the GPU syncs itself.

```cpp
class GpuWavefrontRaytracer {
public:
    explicit GpuWavefrontRaytracer(Context& vkCtx, const Scene& scene);
    ~GpuWavefrontRaytracer();

    // Rebuild GPU state from scene (BLAS, TLAS, textures, queues).
    // Called once after construction and on scene changes.
    void setup(const Scene& scene, uint32_t width, uint32_t height);

    // Called every frame from main thread.
    // Launches wavefront kernels onto stream, signals semaphore, returns immediately.
    // The CPU NEVER blocks inside this function.
    void render(uint32_t spp);

    // Returns the buffer and timeline value Vulkan must wait for.
    FrameInfo getFrameInfo() const;

private:
    Context&            vkCtx;
    const Scene&        scene;
    WavefrontQueues     queues;
    SceneTraversable    tlas;

    // CUDA / OptiX state
    OptixDeviceContext  optixCtx{};
    OptixPipeline       pipeline{};
    OptixShaderBindingTable sbt{};

    // Interop: ping-pong AOV sets
    std::array<std::vector<cudaExternalMemory_t>, 2> aovExtMem;
    std::array<std::vector<cudaSurfaceObject_t>, 2>  aovSurfaces;
    cudaExternalSemaphore_t renderReady{};
    cudaExternalSemaphore_t bufferReleased{};

    cudaStream_t         stream{}; // nonblocking, highest available priority
    cudaTextureObject_t* textureObjects{};  // device-local, CUDA-only

    uint32_t             nextBuffer{0};     // which set CUDA writes next
    uint32_t             lastLaunched{0};
    uint64_t             lastReadyValue{0};
    uint64_t             submittedFrame{0};
    std::array<uint64_t, 2> lastUseValue{};

    void launchExtend(uint32_t depth);
    void launchConnect(uint32_t depth);
    void launchKernel(void* fn, dim3 grid, dim3 block, void** args);
};
```

`render()` — the entire path tracing frame, non-blocking:

```cpp
void GpuWavefrontRaytracer::render(uint32_t spp) {
    uint32_t buf = nextBuffer;
    uint64_t frameValue = ++submittedFrame;

    if (lastUseValue[buf] != 0)
        waitForBufferRelease(lastUseValue[buf]); // async wait on CUDA stream

    for (uint32_t s = 0; s < spp; ++s) {
        // One asynchronous clear for rayCounts[0..maxBounces].
        clearRayCounts();
        launchKernel(generate_kernel, buf);

        for (uint32_t bounce = 0; bounce < maxBounces; ++bounce) {
            launchExtend(bounce);    // direct-index input -> direct-index hit
            launchKernel(shade_kernel, ...);
            launchConnect(bounce);   // direct-index shadow visibility
        }

        launchKernel(finalize_kernel, buf);  // writes into AOV set [buf]
    }

    signalRenderReady(frameValue); // async signal on CUDA stream

    lastLaunched = buf;
    lastReadyValue = frameValue;
    lastUseValue[buf] = frameValue;
    nextBuffer   = 1 - buf;

    // Return immediately — CPU is free
}
```

Vulkan v-sync consumes the frame:

```cpp
FrameInfo GpuWavefrontRaytracer::getFrameInfo() const {
    return {lastLaunched, lastReadyValue, renderReadyVkSemaphore};
}
```

The Vulkan submission waits for `FrameInfo::readyValue`, samples `FrameInfo::bufferIndex`, and signals the same value on `bufferReleasedVkSemaphore` after the tonemapper pass completes.

**Lifecycle:**

| Concern | How it's handled |
|---------|-----------------|
| CPU never blocks | `cudaLaunchKernel`, `optixLaunch`, `cudaSignalExternalSemaphoresAsync` are all non-blocking stream operations |
| GPU sync | Vulkan waits on `renderReady`; CUDA waits on `bufferReleased` before buffer reuse |
| Queue races | Per-depth counters are cleared once before any producer runs; one-to-one hit/shadow slots have one writer and one later reader |
| OptiX parameter races | Every pass/depth launch uses its own persistent device parameter slot and immutable SBT while in flight |
| Atomic pressure | No consumer, hit, or shadow atomics; at most one ray-queue reservation per active warp in Generate or Shade |
| Buffer reuse | Each AOV buffer records its last timeline value and cannot be reused until Vulkan signals that value on `bufferReleased` |
| First frame | `render()` is called once before entering the v-sync loop (blocking, one-time). After that, all launches are async |
| Scene changes | `setup()` rebuilds TLAS/textures synchronously (blocking, but rare — only on user interaction). `render()` is called repeatedly after that |

---

## Phase 8: CUDA Device Selection

```cpp
// src/GPU/CudaDevice.h
int select_cuda_device_for_vulkan(VkPhysicalDevice physicalDevice);
std::string cuda_device_name(int deviceIndex);
```

```cpp
// Compare VkPhysicalDeviceIDProperties::deviceUUID against
// cudaDeviceProp::uuid for every CUDA device, then cudaSetDevice(match).
```

CUDA and Vulkan must select the same physical RTX 4070 before importing memory or semaphores. `NoorRay::init()` performs this match and then constructs `GpuWavefrontRaytracer` unconditionally. No runtime renderer fallback exists.

---

## Phase 9: CMake Build System

```cmake
# Root CMakeLists.txt

# Raise the project minimum to a CUDA-capable CMake baseline and make CUDA required.
cmake_minimum_required(VERSION 3.25)
project(NoorRay VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit 13.3 REQUIRED)
set(OPTIX_ROOT "/home/marcel/Programs/OptixSDK" CACHE PATH "OptiX 9.1 headers")
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" REQUIRED)
file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" OPTIX_VERSION_LINE
     REGEX "^#define OPTIX_VERSION 90100$")
if(NOT OPTIX_VERSION_LINE)
    message(FATAL_ERROR "NoorRay requires the installed OptiX 9.1 headers")
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

file(GLOB_RECURSE NR_KERNEL_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/Kernels/*.h"
    "${PROJECT_SOURCE_DIR}/src/GPU/Annotations.h"
    "${PROJECT_SOURCE_DIR}/src/GPU/RayTraversal.h")

# OptiX device programs are not normal CUDA objects. Build one OptiX-IR blob,
# then generate a C++ byte-array header in the build tree.
set(NR_OPTIX_IR "${CMAKE_CURRENT_BINARY_DIR}/generated/NoorRayOptix.optixir")
set(NR_OPTIX_IR_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/NoorRayOptixIr.h")
add_custom_command(
    OUTPUT "${NR_OPTIX_IR}" "${NR_OPTIX_IR_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            --optix-ir --gpu-architecture=compute_89 --std=c++20
            --relocatable-device-code=true --use_fast_math --generate-line-info
            -DNR_BACKEND_CUDA=1 -DGLM_FORCE_CUDA=1
            "-I${PROJECT_SOURCE_DIR}/src"
            "-I${PROJECT_SOURCE_DIR}/external/glm"
            "-I${OPTIX_INCLUDE_DIR}"
            -o "${NR_OPTIX_IR}"
            "${PROJECT_SOURCE_DIR}/src/Kernels/cuda/OptixPrograms.cu"
    COMMAND "${CMAKE_COMMAND}"
            -DBIN2C="${CUDAToolkit_BIN_DIR}/bin2c"
            -DINPUT="${NR_OPTIX_IR}"
            -DOUTPUT="${NR_OPTIX_IR_HEADER}"
            -P "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    DEPENDS
            src/Kernels/cuda/OptixPrograms.cu
            ${NR_KERNEL_HEADERS}
            cmake/EmbedBinary.cmake
    VERBATIM)
add_custom_target(NoorRayOptixIr DEPENDS "${NR_OPTIX_IR_HEADER}")

add_library(NoorRayCuda STATIC
    src/GPU/CudaDevice.cpp
    src/GPU/Memory.cpp
    src/GPU/ImageInterop.cpp
    src/Kernels/cuda/Kernels.cu
    src/Kernels/cuda/AccelBuilder.cu
    src/Kernels/cuda/OptiXSetup.cu
    src/Raytracing/GpuWavefrontRaytracer.cpp
    "${NR_OPTIX_IR_HEADER}")
add_dependencies(NoorRayCuda NoorRayOptixIr)
set_target_properties(NoorRayCuda PROPERTIES CUDA_ARCHITECTURES 89)
target_compile_features(NoorRayCuda PUBLIC cxx_std_20 cuda_std_20)
target_compile_definitions(NoorRayCuda PUBLIC NR_BACKEND_CUDA=1)
target_compile_definitions(NoorRayCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:GLM_FORCE_CUDA=1>)
target_include_directories(NoorRayCuda PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/external/glm"
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(NoorRayCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--generate-line-info;--extended-lambda>)
target_link_libraries(NoorRayCuda PUBLIC CUDA::cudart CUDA::cuda_driver glm Vulkan::Vulkan)

# The existing broad source glob must explicitly exclude CUDA implementation
# files and all retained Vulkan renderer reference implementations. Their headers
# remain in the repository, but no old renderer object code enters NoorRay.
list(FILTER SRC_FILES EXCLUDE REGEX "/GPU/")
list(FILTER SRC_FILES EXCLUDE REGEX "/Kernels/cuda/")
list(FILTER SRC_FILES EXCLUDE REGEX "/Raytracing/(Raytracer|WavefrontRaytracer|ComputeRaytracer|RtxRaytracer|GpuWavefrontRaytracer|RayLutGenerator)\\.cpp$")

target_link_libraries(NoorRay PRIVATE NoorRayCuda)
target_compile_definitions(NoorRay PRIVATE NR_BACKEND_CUDA=1)
```

`cmake/EmbedBinary.cmake` invokes `bin2c --const --static --length --name noorRayOptixIr` with `OUTPUT_FILE`, then adds `#pragma once`; no shell redirection is used. `OptixPrograms.cu` is only an input to the IR custom command and must never appear in `NoorRayCuda` sources.

In the real root file, create/filter `SRC_FILES` before `add_executable(NoorRay ...)`. Replace `NoorRayShaders` with a tonemapper-only custom target; retained wavefront Slang and SPIR-V reference files are not regenerated. There is no CUDA/HIP option matrix: CUDA is required for this Linux build.

---

## What Is Removed from Vulkan

| Item | Status |
|------|--------|
| Existing Vulkan renderer `.h/.cpp` and Slang files | **Retained as reference, excluded from build, never called** |
| Vulkan BLAS/TLAS creation in active scene setup | **Removed from active code**; OptiX owns all acceleration structures |
| Ray-query device extensions in active Vulkan startup | **Not requested** |
| Wavefront SPIR-V compilation | **Removed from CMake**; existing sources/blobs remain as reference |
| Slang compiler | **Retained only for the Vulkan tonemapper** |
| Mesh GPU ownership | CPU mesh vectors remain authoritative; `GpuWavefrontRaytracer` uploads CUDA copies and builds OptiX BLASes |

Vulkan retains the swapchain, external memory/semaphore extensions, ImGui, tonemapper, and final display. `MeshAsset` stops allocating Vulkan geometry buffers, Vulkan BLASes, or software BVHs in its active construction path; the retained renderer sources are historical reference code rather than a compilable fallback.

---

## Locked Feature-Parity Contract

The CUDA renderer is not considered complete until it preserves all currently reachable behavior:

- Perspective, orthographic, fisheye, thin-lens, and realistic cameras.
- Realistic-camera ray-LUT loading, invalidation, regeneration, and cache behavior.
- R2 and Halton sampling, sample/frame indexing, adaptive sampling, and accumulation/reset semantics.
- Existing PBR material/BXDF behavior, texture addressing/filtering, normal data, and face-material selection.
- Directional, environment, and scene-light sampling with one NEE shadow ray per active path.
- Diffuse, specular, transmission, hit, and maximum-depth termination counters and Russian roulette behavior.
- Color, albedo, normal, cryptomatte, position, and material AOV values and formats.
- Mesh creation/import, instance transforms, material edits, camera edits, resize, scene reload, and accumulation invalidation.
- Existing UI controls and output selection. The UI uses `GpuWavefrontRaytracer` exclusively.

The retained Vulkan implementation is the semantic reference. It is not linked into NoorRay and cannot be selected at runtime.

---

## One-Pass Implementation Sequence

### Step 1 — Build Foundation
1. Raise CMake minimum, enable required CUDA, find CUDA 13.3 and OptiX 9.1, target `sm_89`.
2. Replace broad shader generation with tonemapper-only Slang generation.
3. Add explicit source exclusions for retained renderer references and CUDA implementation sources.
4. Add OptiX-IR generation, `bin2c` embedding, and dependency tracking for every kernel header.
5. Add CUDA device selection by Vulkan UUID and compile a minimal OptiX pipeline.

### Step 2 — Scene and OptiX
1. Implement `OptiXSetup.cu` using the embedded OptiX IR and hit-object traversal.
2. Implement `AccelBuilder.cu` — BLAS from device-local vertex/index data, TLAS with instances
3. Upload all scene/material/texture data into device-local CUDA storage.
4. Implement BLAS rebuild and TLAS update paths for geometry and transform edits.

### Step 3 — Complete Kernel Port
1. Port the five passes (`Generate`, `Extend`, `Shade`, `Connect`, `Finalize`) to `src/Kernels/passes/*.h`
2. Write `Kernels.cu` launchers for non-RT passes
3. Write the combined `OptixPrograms.cu` Extend/Connect raygen programs.
4. Implement `GpuWavefrontRaytracer` — queue allocation, launch loop
5. Port every feature in the locked feature-parity contract; do not stop at first light transport.

### Step 4 — Vulkan Interop and UI
1. Implement 12 format-correct export/import AOV images and two bidirectional timeline semaphores.
2. Update texture loading to allocate `cudaArray_t` + `cudaTextureObject_t` directly (no Vulkan)
3. Verify zero-copy: GPU-side perf capture shows no AOV staging transfers
4. Replace active renderer construction and all UI/output access with `GpuWavefrontRaytracer`.

### Step 5 — Verification and Cleanup
1. Build with warnings enabled and run layout/static assertions for every GPU ABI type.
2. Run deterministic low-resolution image checks for every camera, sampler, material/light path, and AOV.
3. Run `compute-sanitizer` racecheck/memcheck on reduced scenes and stress queue capacity at maximum depth.
4. Profile with Nsight Systems/Compute; record pass timings and total frame latency on the RTX 4070.
5. Remove all active references to old renderer constructors and Vulkan ray-tracing extensions while retaining their source files unchanged as reference where possible.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| CUDA/OptiX version drift | Low | CMake requires CUDA 13.3 and verifies `OPTIX_VERSION == 90100` |
| `cudaExternalMemoryGetMappedMipmappedArray` format mismatch | Medium | Use the exact per-AOV CUDA channel descriptor table and dedicated Vulkan allocations |
| Vulkan/CUDA select different GPUs | Low | Match Vulkan and CUDA UUIDs before importing any external object; fail setup if no match exists |
| Timeline semaphore ownership error | Medium | Two directional timeline semaphores; CUDA waits for release before reuse and Vulkan waits for ready before sampling |
| `glm::vec3` + `GLM_FORCE_CUDA` pulls unused GLM code into kernel TUs | Low | Only include needed GLM headers (`vec3.hpp`, `vec4.hpp`, `mat4x4.hpp`); no full `<glm/glm.hpp>` in kernel code |
| OptiX launch width exceeds the active count at deep bounces | Medium | Launch the fixed queue capacity and return when `idx >= rayCounts[depth]`; never read the count back to the CPU. Profile capacity-sized launches and render in tiles if sparse-launch cost becomes material. |
| Continuation queue overflow | Low | Capacity equals the maximum active sample count; each active path emits at most one continuation, and debug builds trap if a reservation exceeds capacity. |
| Accidental same-depth path-state aliasing | Low | Preserve one active ray and at most one shadow ray per sample; assert this invariant in debug validation kernels. |
| Async OptiX launch-parameter overwrite | Low | Persistent per-depth Extend/Connect parameter arrays; update only after synchronized lifecycle changes. |
| Retained Vulkan reference code becomes stale | Expected | Exclude it from all targets and treat it as read-only semantic reference, not a supported fallback. |
| Feature parity regression | Medium | Maintain deterministic reduced-resolution checks covering the locked parity contract before performance tuning. |
