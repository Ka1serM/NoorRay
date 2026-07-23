# hdNoorRay

`hdNoorRay` is NoorRay's OpenUSD Hydra render-delegate plugin. OpenUSD types
are confined to this directory; `libnoorray` remains independent of USD.

## Compatibility

The target is Blender 5.2, which creates a legacy `HdRenderDelegate` through
`HdRendererPluginRegistry::CreateRenderDelegate`. OpenUSD can also adapt this
delegate into its Hydra 2 renderer path.

The plugin must be compiled against the exact OpenUSD ABI of its host. For
Blender, point `NR_USD_ROOT` at an SDK assembled from that Blender build's
dependency package, containing:

```text
include/pxr/pxr.h
lib/libusd_ms.so
```

Blender 5.2 development builds used OpenUSD 25.08 and the current 5.2 release
branch uses OpenUSD 26.03. The adapter supports both source APIs, but each
binary is tied to the OpenUSD version and internal namespace of the Blender
build it was compiled for.

Configure NoorRay with:

```sh
cmake -S . -B build \
  -DNR_BUILD_HYDRA=ON \
  -DNR_USD_ROOT=/path/to/blender-5.2-usd-sdk \
  -DNR_CUDA_ROOT=/usr/local/cuda \
  -DOPTIX_ROOT=/path/to/OptixSDK
```

The plugin bundle is written to `build/hdnoorray/plugin`. Register that
directory with `pxr.Plug.Registry().RegisterPlugins()` in Blender, or add it to
`PXR_PLUGINPATH_NAME` for `usdview`.

The build also creates
`build/hdnoorray/hdnoorray-blender-5.2.zip`. Install that archive through
Blender's Extensions preferences. The extension bundles the compiled delegate,
registers `HdNoorRayRendererPlugin`, and adds **NoorRay** to the render-engine
selector.

The first viewport implementation supports polygon meshes (fan-triangulated),
object transforms, perspective/orthographic cameras, a default diffuse
material, and color/depth AOVs. Material networks, authored normals/UVs,
textures, lights, and Hydra instancing remain follow-up work.
