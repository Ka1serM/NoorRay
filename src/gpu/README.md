# NoorRay GPU API

`noorray_gpu` is the small Vulkan-backed API described by
the Vulkan GPU API implementation plan. Its public headers
are under `src/gpu/include/gpu` and intentionally contain no Vulkan or VMA
types.

The current implementation provides:

- typed `Buffer<T>` objects backed by stable buffer-device addresses;
- an explicit descriptor-backed `Buffer<T>::handle()` for APIs that require a
  resource token, while ordinary shader access remains BDA-oriented;
- staging uploads and timeline-based readbacks;
- precompiled SPIR-V shader creation;
- root-argument compute dispatch and indirect dispatch address lookup;
- high-level barriers, timeline tokens, and deferred command/resource lifetime;
- opaque image and sampler handles, with `GENERAL` image layout policy;
- explicit high-level image formats, including separate sampled and storage
  handles when one image is used for both roles;
- descriptor-heap-backed image and sampler descriptors when
  `VK_EXT_descriptor_heap` is available;
- dynamic-rendering graphics pipelines and render targets;
- capability-gated BLAS/TLAS construction, in-place TLAS refit, and
  ray-tracing pipelines with internally owned shader-binding tables;
- range-based uploads for updating persistent scene buffers without replacing
  their GPU addresses or descriptor-heap slots;
- device timestamp queries used by NoorRay's `--stats` output.

The GPU API tests and examples compile Slang sources to SPIR-V with `slangc`.
Their shaders use GPU pointers and descriptor-heap handles, with only the root
argument pointer passed as a push constant; they contain no resource bindings.
The library is linked into `libnoorray` and owns the shared VMA implementation,
so NoorRay and the standalone GPU tests use one backend allocation layer. The
NoorRay's ray-tracing and viewport passes use the same API-owned descriptor heap,
resource-heap pipelines, and external-command recording bridge; only the
swapchain-facing command sequencing and legacy image readback remain in the
application boundary.

Shaders are always supplied as bytes. Shader compilation and resource
reflection are outside this library.
