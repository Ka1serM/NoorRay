# hdNoorRay

`hdNoorRay` is NoorRay's OpenUSD Hydra render-delegate plugin for Blender 5.2.
OpenUSD types are confined to this directory; `libnoorray` remains independent
of USD.

## Compatibility

The target is Blender 5.2, which creates a legacy `HdRenderDelegate` through
`HdRendererPluginRegistry::CreateRenderDelegate`. OpenUSD can also adapt this
delegate into its Hydra 2 renderer path.

The plugin must be compiled against the exact OpenUSD ABI of its host. CMake
automatically vendors the USD SDK from Blender's dependency package when
`NR_USD_ROOT` is left at its default.

## Prerequisites

- Blender 5.2 installed (provides `libusd_ms.so`)
- `python3.13-devel` — the vendored USD Python headers only ship
  `pyconfig.h`; the rest are pulled from the system's Python 3.13
- NVIDIA CUDA Toolkit 13.x at `/usr/local/cuda`
- OptiX 9.1 at `$HOME/Programs/OptixSDK`
- GCC/G++ 15 with C++23 support
- Vulkan SDK (set `VULKAN_SDK`)

## Build

```sh
cmake -S . -B build/hdnoorray \
  -DNR_BUILD_HYDRA=ON \
  -DNR_CUDA_ROOT=/usr/local/cuda \
  -DOPTIX_ROOT="$HOME/Programs/OptixSDK" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DNR_CUDA_ARCH='86;89' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build/hdnoorray --target hdNoorRay -j"$(nproc)"
```

The vendored USD SDK is automatically fetched from Blender's
`lib-linux_x64` repository on first configure. The Blender extension is
assembled at `build/hdnoorray/hdnoorray/blender_extension/hdnoorray/`.

## Install in Blender

Link the built extension into Blender's user extension directory:

```sh
mkdir -p ~/.config/blender/5.2/extensions/user_default
ln -sfn "$PWD/build/hdnoorray/hdnoorray/blender_extension/hdnoorray" \
  ~/.config/blender/5.2/extensions/user_default/hdnoorray
```

Restart Blender and select **NoorRay** from the render-engine dropdown.

## Status

The first viewport implementation supports polygon meshes (fan-triangulated),
object transforms, perspective/orthographic cameras, a default diffuse
material, and color/depth AOVs. Material networks, authored normals/UVs,
textures, lights, and Hydra instancing remain follow-up work.
