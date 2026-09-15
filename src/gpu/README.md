# gpu

A small, object-oriented Vulkan API. Buffers are device addresses; textures and
samplers are 32-bit indices into descriptor heaps the device owns and fills
automatically. There are no pipeline layouts, descriptor sets, resource
registries in the app, or upload batches.

Texture binding uses the cross-vendor `VK_EXT_descriptor_heap` model; see
[Requirements and verification](#requirements-and-verification).

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /your/prefix
```

Vulkan and [VulkanMemoryAllocator][vma] are the only dependencies. VMA is found
via `find_package(VulkanMemoryAllocator)`, a vendored copy at
`external/VulkanMemoryAllocator`, or an explicit `-DGPU_VMA_DIR=`.
`GPU_BUILD_EXAMPLES` and `GPU_BUILD_TESTS` default to `ON` for a standalone
build and `OFF` when this project is added as a subdirectory; both need `slangc`
from the Vulkan SDK, and the tests additionally need Catch2 v3.

[vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator

## Consuming

```cmake
find_package(gpu REQUIRED)
target_link_libraries(my_app PRIVATE gpu::gpu)
```

Or vendor it and `add_subdirectory(external/gpu)`, which defines the same
`gpu::gpu` target.

```cpp
gpu::Device device;
auto vertices = device.buffer<Vertex>(source.size());
vertices.upload(std::span<const Vertex>(source));

gpu::Shared<SceneData> scene(device);
scene.data.vertices = vertices.ptr();
scene.commit();

struct FrameData { gpu::GpuPtr<SceneData> scene; Camera camera; };
pipeline.launch(groups, FrameData{scene.ptr(), camera});
```

Buffer, image, sampler, and shared-data owners are move-only. Internal references
keep submitted Vulkan objects alive until completion. GPU pointers and texture
handles are non-owning: keep their owners alive while using them, including
objects referenced indirectly through another GPU record. Destroy resources
before their Device; the surface provider outlives its Device.

## Data and uploads

- `Buffer<T>`: `ptr()`, `upload(span, elementOffset)`, `download(span)`.
- `Image<T>`: full-level `upload(span)` and `download(span)`;
  `sampled_handle()` and `storage_handle()` for shaders.
  `handle()` is a checked host reference for render targets and interop.
- `Sampler`: `handle()` for shaders. There is no combined image/sampler; a
  shader pairs a sampled texture handle with a sampler handle.

## Textures and samplers

The device creates one resource heap and one sampler heap at construction, sized
by `DeviceConfig::texture_descriptor_capacity` (default 16384) and
`DeviceConfig::sampler_descriptor_capacity` (default 256). A `Sampled` or
`Storage` image takes one resource slot per usage, and a sampler takes one
sampler slot. Handles are slot indices, so shaders index the heaps directly:

```slang
RWTexture2D<float4> output = ResourceDescriptorHeap[arguments.output];
Texture2D<float4> albedo = ResourceDescriptorHeap[arguments.albedo];
SamplerState linear = SamplerDescriptorHeap[arguments.sampler];
```

Slot 0 of each heap is reserved and never written, so a zero handle means "no
texture". Store handles as `uint` in shared records. A resource returns its
slots when it is retired, meaning after every submission that might still read
them has finished. Until then its handles stay valid for in-flight work.
Slots are then reused. The heaps never grow: running out throws
`ErrorCode::OutOfMemory`.

The root argument pointer is delivered with `vkCmdPushDataEXT`, and both heaps are
bound before every dispatch, draw, and trace. That makes it safe to record
legacy descriptor-set commands, such as an ImGui backend's, into the same
frame command buffer.
- `Shared<T>`: public `data`, `ptr()`, `commit()`. Commit compares the
  trivially-copyable record and skips unchanged bytes.

Upload copies into temporary mapped memory before returning. The caller may
immediately reuse its source; staging lives until the GPU copy finishes.
Temporary duplicate memory is intentional. Download waits for its copy.
There is no fixed staging budget or global dirty-object list.

Small per-frame values, including NoorRay's camera, go directly in root arguments.
The backend copies these into a mapped arena and retires ranges by timeline value.
Exhausting the arena within one open frame throws instead of deadlocking; increase
`DeviceConfig::argument_arena_bytes` if a frame genuinely needs more space.

## Frames and synchronization

Prepare changed resources and commit shared data **before** `begin_frame()`.
Launch rendering and presentation between begin/end; they share one submission.
End presents without a device-wide CPU wait. Acquire slots wait only when reused.
Timestamp reads are nonblocking and return the latest available measurement.

Uploads/readbacks during an open frame are rejected. Large scene replacement,
resize, and destruction may synchronize. Ordered submissions handle normal
updates and TLAS refits. Dropping a Frame discards it and rebuilds the acquired
swapchain before the next frame. The window provider reports live pixel dimensions.

The API is single-threaded per Device. Raw pointers cannot validate arbitrary
shader pointer graphs; callers remain responsible for pointer bounds, compatible
host/shader layouts, and owner lifetime. Vulkan interop additionally requires the
external API to finish using exported objects before their owners are destroyed.

## Requirements and verification

Vulkan 1.3 with buffer device addresses, timeline semaphores, synchronization2,
dynamic rendering and unified image layouts is mandatory, as are
`VK_EXT_descriptor_heap` (`descriptorHeap`), `VK_KHR_shader_untyped_pointers`
(`shaderUntypedPointers`) and maintenance5 (core in 1.4, or the extension).
Devices without them are skipped at selection. None of these is vendor-specific,
but driver support is recent: at the time of writing, NVIDIA's driver exposes
`VK_EXT_descriptor_heap`, and AMD and Intel need a driver version that ships it.
Ray tracing is checked separately.

Shaders are supplied as SPIR-V bytes. Any shader that indexes
`ResourceDescriptorHeap` or `SamplerDescriptorHeap` must be compiled with
`slangc -capability spvDescriptorHeapEXT`, and the heap stride must stay at its
default (do not pass `-spirv-resource-heap-stride`). Validate modules with
`spirv-val --target-env vulkan1.4`. Requested validation fails clearly if its
layer is absent.

`memory_report()` reports VMA allocation bytes, reserved block bytes, allocation
count, and argument-arena capacity, including pending allocations. Dedicated
external-memory images are outside these VMA totals.

GPU tests cover compute, images (including heap reads and writes, and slot reuse
and exhaustion), ray tracing, shared updates, argument wraparound, and retirement. `gpu_presentation_test` additionally needs a desktop display and
tests resizing and abandoned frames. Enable synchronization validation with
`VK_LAYER_VALIDATE_SYNC=1`.
