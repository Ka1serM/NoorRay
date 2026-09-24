# MaterialX in NoorRay

NoorRay compiles MaterialX documents and Blender-exported MaterialX graphs to
Slang with MaterialX's Slang generator, then to SPIR-V callable shaders that the
realtime renderer invokes per hit.

## Data flow

Standalone `.mtlx` files and Hydra materials follow the same path:

    MaterialX document
        -> SlangMaterialGenerator (MaterialXGenSlang, NoorRay target)
        -> Slang module implementing IMaterial + parameter block
        -> SlangMaterialCompiler (Slang library)
        -> SPIR-V callable shader

Pattern nodes keep their MaterialX genslang implementations; BSDF, EDF and
surface nodes build the closures of `MaterialInterface.slang` instead of lighting
the surface. Every editable input lives in the parameter block, so documents
that differ only in values or texture bindings share one compiled shader.

The MaterialX libraries are embedded in the binary and unpacked to the user
cache on first use, because MaterialX reads definitions and generator sources
only from files.

## Blender integration

The Blender extension exports original Blender node names into a MaterialX
document and preserves links, defaults, color/vector widths, normals, and
shader mixing. The exporter reports every reached Blender node without a
semantic handler instead of silently claiming complete support. Its coverage
is tracked in [Blender exporter coverage](Blender_MaterialX_Exporter_Coverage.md).

## Current boundaries

The spectral path tracer does not evaluate materials yet: it shades every
surface as a neutral diffuse until it runs the same compiled shaders.

Documents with no renderable surface, or with closures the realtime renderer
cannot express, fail generation and fall back to the default material.
