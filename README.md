## NoorRay 

My personal path tracer for exploring graphics programming and testing out new rendering techniques.

<img width="962" height="563" alt="image" src="https://github.com/user-attachments/assets/3a4894e1-7d78-478f-8da2-89a528f2d4d7" />

### Features

- **Vulkan Backend:** Implements the interactive path tracer and viewport with Vulkan and Slang compute/ray-query shaders.

- **Path Tracing with MIS:** Uses **unidirectional path tracing** with **multiple importance sampling (MIS)**. Supports combined **Lambertian diffuse** and **GGX specular** materials.  

- **Scene Loading:** Uses USD (`.usd`, `.usda`, and `.usdc`) as the native scene format, while continuing to read legacy NoorRay `.nrscene`, PBRT v4 `.pbrt`, Wavefront `.obj`, and Khronos `.gltf`/`.glb` files.

- **Material System:** Full **Disney PBR** support with **albedo, roughness, metallic, normal, transmission, opacity, and emission**.  

- **ImGui Interface:** Provides a user interface to edit scene parameters, including camera settings, scene graph, and material properties in real-time.  

- **Platform:** The current backend targets Linux with a Vulkan-capable GPU.


### Build Instructions

#### Prerequisites

- **Vulkan SDK:** Set `VULKAN_SDK` to the active SDK installation.
- **CMake:** Version 3.25 or newer.
- **Compiler:** clang/clang++ 22 with C++23 support, passed once as
  `CMAKE_CXX_COMPILER` (CMake requires it explicitly rather than defaulting).
- **Python bindings:** `uv` and Python 3.12.
- **MaterialX:** the vendored MaterialX standard library is compiled as part
  of the NoorRay build (see `docs/MaterialX.md`).


#### Clone the Repository

Clone the repository including its submodules:

```bash
git clone --recursive git@github.com:Ka1serM/NoorRay.git
cd NoorRay
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

#### Python Environment

Create the Python 3.12 environment used by both CMake and the run tasks:

```bash
rm -rf .venv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r src/pynoorray/requirements.txt
```

Passing `Python_EXECUTABLE` during configuration is required. The native module
must be built for the same CPython ABI that runs it.

#### Configure

The following release configuration builds for NVIDIA RTX 30-series (`sm_86`)
and RTX 40-series (`sm_89`) GPUs:

```bash
cmake -S . -B build/release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-22 \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNR_BUILD_PYTHON=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Use `build/debug` and `-DCMAKE_BUILD_TYPE=Debug` for a debug configuration.
The equivalent commands are also available as Zed tasks.

#### Build and Run

Build and run the application:

```bash
cmake --build build/release --target NoorRay -j"$(nproc)"
./build/release/NoorRay
```

Build and run the test suite:

```bash
cmake -S . -B build/release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-22 \
  -DNR_BUILD_TESTS=ON
cmake --build build/release -j"$(nproc)"
ctest --test-dir build/release --output-on-failure
```

Tests are enabled by default. Set `NR_BUILD_TESTS=OFF` for a production-only build.

### Shader Compilation

Shaders are compiled automatically by CMake using `slangc` from the Vulkan SDK. No separate shader recompilation script is needed; building the `NoorRay` target regenerates the SPIR-V files when shader sources change.


### Dependencies

This project uses the following libraries (included as submodules):

-   SDL3
-   Dear ImGui
-   GLM
-   portable-file-dialogs
-   tinyobjloader
-   tinygltf
-   stb\_image
