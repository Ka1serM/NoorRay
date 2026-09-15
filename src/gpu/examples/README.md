# gpu API examples

These small programs use the public `gpu` API directly and compile their
shaders with Slang, matching NoorRay's production shader toolchain. They do
not create a NoorRay `Context` or use the renderer:

```sh
cmake -S . -B build -DNR_BUILD_EXAMPLES=ON
cmake --build build --target gpu_compute_example gpu_graphics_example gpu_raytracing_example
./build/examples/gpu_api/gpu_compute_example
./build/examples/gpu_api/gpu_graphics_example
./build/examples/gpu_api/gpu_raytracing_example
```

The compute example uploads two arrays and checks their sum. The graphics
example records a triangle into an off-screen color attachment. The ray
tracing example builds a BLAS and TLAS, launches a ray-generation shader, and
checks its output. The shaders contain no descriptor or layout bindings; only
the root pointer supplied by `gpu::Device` is used for compute and ray tracing.
Ray tracing reports a skip when the selected Vulkan device does not expose the
required feature.
