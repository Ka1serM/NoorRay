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
- Python development headers matching the Blender USD SDK
- clang/clang++ 22 with C++23 support, passed as `CMAKE_CXX_COMPILER`
- Vulkan SDK (set `VULKAN_SDK`)
- MaterialX ingestion (always enabled): see the root `README.md`

## Build

```sh
cmake -S . -B build/hdnoorray \
  -DNR_BUILD_HYDRA=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-22 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build/hdnoorray --target hdnoorray -j"$(nproc)"
```

The vendored USD SDK is automatically fetched from Blender's
`lib-linux_x64` repository on first configure. The Blender extension is
assembled at `build/hdnoorray/blender_extension/hdnoorray/`.

## Install in Blender

Link the built extension into Blender's user extension directory:

```sh
mkdir -p ~/.config/blender/5.2/extensions/user_default
ln -sfn "$PWD/build/hdnoorray/blender_extension/hdnoorray" \
  ~/.config/blender/5.2/extensions/user_default/hdnoorray
```

Restart Blender and select **NoorRay** from the render-engine dropdown.

## Status

The Blender delegate supports polygon meshes and instancing, generated or
authored normals/UVs, MaterialX surface graphs and textures, analytic and dome
lights, progressive viewport rendering, offline Combined output, and NoorRay's
camera and Gaussian-splat extensions.

## Texture transport

Hydra gives render delegates asset paths, not Blender `ImBuf` pointers.
hdNoorRay opens those assets through OpenUSD Hio and copies the decoded pixels
straight into NoorRay-owned memory. Loads are shared for one Hydra sync batch;
materials and the environment then retain only the textures they actually use.
This batch lifetime is important because Blender can rewrite a generated image
at the same cache path during interactive edits.

External image files that Hio supports are passed through at their existing
path. Blender itself writes generated images, packed images, and formats it must
convert to a session cache before building the Hydra material network. Avoiding
that upstream cache would require a Blender image-memory/resolver hook; the
public `HydraRenderEngine` interface currently exposes only filename-based
assets.
