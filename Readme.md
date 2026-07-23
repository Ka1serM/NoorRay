## NoorRay 

My personal path tracer for exploring graphics programming and testing out new rendering techniques.

<img width="962" height="563" alt="image" src="https://github.com/user-attachments/assets/3a4894e1-7d78-478f-8da2-89a528f2d4d7" />

### Features

- **CUDA/OptiX Backend:** Implements an OptiX path tracer with CUDA 13 and OptiX 9.1, with Vulkan/CUDA external-memory interop for the interactive viewport.

- **Path Tracing with MIS:** Uses **unidirectional path tracing** with **multiple importance sampling (MIS)**. Supports combined **Lambertian diffuse** and **GGX specular** materials.  

- **Scene Loading:** Supports NoorRay `.nrscene`, PBRT v4 `.pbrt`, Wavefront `.obj`, and Khronos `.gltf`/`.glb` files.

- **Material System:** Full **Disney PBR** support with **albedo, roughness, metallic, normal, transmission, opacity, and emission**.  

- **ImGui Interface:** Provides a user interface to edit scene parameters, including camera settings, scene graph, and material properties in real-time.  

- **Platform:** The current backend targets Linux with an NVIDIA GPU. The former macOS and Windows release paths were removed when CUDA/OptiX became mandatory.

### Build Instructions

#### Prerequisites

- **Vulkan SDK:** Set `VULKAN_SDK` to the active SDK installation.
- **CUDA Toolkit:** CUDA 13.x installed at `/usr/local/cuda`, with the compiler
  at `/usr/local/cuda/bin/nvcc`.
- **OptiX SDK:** OptiX 9.1 installed at `$HOME/Programs/OptixSDK`, with headers
  under `$HOME/Programs/OptixSDK/include`.
- **CMake:** Version 3.25 or newer.
- **Compiler:** GCC/G++ 15 with C++23 support. CUDA compilation also uses G++ 15
  as its host compiler.
- **Python bindings:** `uv` and Python 3.12.


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
  -DNR_CUDA_ROOT=/usr/local/cuda \
  -DOPTIX_ROOT="$HOME/Programs/OptixSDK" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNR_CUDA_ARCH='86;89' \
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

Build and run the Python example:

```bash
cmake --build build/release --target _pynoorray -j"$(nproc)"
PYTHONPATH="$PWD/build/release/lib" \
  .venv/bin/python src/pynoorray/examples/pynoorray_render.py
```

The extension is copied to `build/release/lib/pynoorray/` and its filename must
contain the `cpython-312` ABI suffix when used with this environment.

Build and run the test suite:

```bash
cmake -S . -B build/release \
  -DNR_CUDA_ROOT=/usr/local/cuda \
  -DOPTIX_ROOT="$HOME/Programs/OptixSDK" \
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
