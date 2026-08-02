# MaterialX in NoorRay

NoorRay compiles MaterialX documents and Blender-exported MaterialX graphs
directly to shared SVM bytecode. The fixed SVM interpreter evaluates that
bytecode on CPU and CUDA; the graph does not create a per-material GPU program.

## Data flow

Standalone `.mtlx` files and Hydra materials follow the same path:

    MaterialX document
        -> node-graph normalization and nodegraph flattening
        -> SvmCompiler
        -> SVM bytecode and texture-index table
        -> SvmEval
        -> MaterialEvaluation and NoorRayCompositeBsdf

The compiler resolves value nodes, transforms, textures, normals, arithmetic,
conditionals, and surface closures. Closure leaves are accumulated into the
fixed composite BSDF, with opacity and emission carried as terminal material
outputs.

## Blender integration

The Blender extension exports original Blender node names into a MaterialX
document and preserves links, defaults, color/vector widths, normals, and
shader mixing. The exporter reports every reached Blender node without a
semantic handler instead of silently claiming complete support.

The supported node catalog is generated from the installed MaterialX libraries
and the NoorRay extensions. Coverage reports are maintained in:

* [MaterialX SVM coverage](MaterialX_SVM_Coverage.md)
* [Blender exporter coverage](Blender_MaterialX_Exporter_Coverage.md)

## Current boundaries

The SVM path is intended for regular surface node graphs. Volume transport,
light-source and emission-distribution graphs, and other features that need
integrator-level transport changes remain explicitly outside the surface
compiler until their runtime contracts are implemented.

Unsupported nodes fail compilation with their MaterialX category. Blender
export validation fails when a reached Blender node has no exporter handler.

## Verification

Focused unit tests cover compiler dispatch, value widths, matrix operations,
normal-map convention, texture alpha, and closure accumulation. The Blender
validation script checks node reachability, XML conversion, and exporter
coverage; the fidelity harness compares SVM renders with an independent
MaterialX reference render.
