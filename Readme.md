## NoorRay 

My personal path tracer for exploring graphics programming and testing out new rendering techniques.

<img width="962" height="563" alt="image" src="https://github.com/user-attachments/assets/3a4894e1-7d78-478f-8da2-89a528f2d4d7" />

### Features

- **Multiple Backends:** Implements path tracing with both **Vulkan hardware-accelerated ray tracing** (via the official Vulkan ray tracing extension) and a compute-shader fallback.  

- **Path Tracing with MIS:** Uses **unidirectional path tracing** with **multiple importance sampling (MIS)**. Supports combined **Lambertian diffuse** and **GGX specular** materials.  

- **Scene Loading:** Supports loading scenes from Wavefront `.obj` files and Khronos `.gltf` files.  

- **Material System:** Full **Disney PBR** support with **albedo, roughness, metallic, normal, transmission, opacity, and emission**.  

- **ImGui Interface:** Provides a user interface to edit scene parameters, including camera settings, scene graph, and material properties in real-time.  

- **Cross-Platform:** Supports **Windows, MacOS, and Linux**.

### Build Instructions


#### Prerequisites

-   **Vulkan SDK:** Ensure the Vulkan SDK is installed and the `VULKAN_SDK` environment variable is set.
-   **CMake:** Version 3.10 or higher.
-   **Compiler:** A C++23-compatible compiler. **MinGW is recommended** (on Windows) due to usage of the `#embed` directive in the source code.


#### Clone the Repository

Clone the repository including its submodules:

```bash
git clone --recursive https://github.com/Ka1serM/VulkanToyPathtracer.git
cd VulkanToyPathtracer
```


#### Building the Project

1.  Create a build directory:

    ```bash
    mkdir build
    cd build
    ```

2.  Configure the project with CMake (MinGW + Release mode):

    ```bash
    cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    ```

3.  Build the project:

    ```bash
    cmake --build . --config Release
    ```

4.  Run the executable:

    The output executable will be located in the `build/` directory.


### Shader Compilation

The shaders are compiled using `glslc` from the Vulkan SDK. A `recompile.bat` script is provided in the `src/Shaders` directory to automate this process.

1.  Navigate to the shader directory:

    ```bash
    cd src/Shaders
    ```

2.  Run the script:

    ```bash
    recompile.bat
    ```

    This script compiles GLSL shaders to SPIR-V bytecode using `glslc`.


### Dependencies

This project uses the following libraries (included as submodules):

-   SDL3
-   Dear ImGui
-   GLM
-   portable-file-dialogs
-   tinyobjloader
-   tinygltf
-   stb\_image
