# SVM (Shading Virtual Machine)

This directory holds NoorRay's MaterialX-to-bytecode compiler and its
interpreter, replacing MdlMaterialCompiler (src/libnoorray/MDL, removed):

- `SvmCompiler.h` -- public entry point (`SvmCompiler::compile`).
- `SvmMaterialXCompiler.cpp` -- walks a `MaterialX::Document` node graph and
  emits SVM bytecode. This is where MaterialX node support lives; see the
  `emitNode`/`emitClosure` category dispatch for the full list of supported
  MaterialX node categories.
- `SvmTypes.h` -- the bytecode instruction set (`NodeType` + per-instruction
  payload structs) shared by the compiler and interpreter.
- `SvmEval.h` -- the interpreter that evaluates compiled bytecode at shading
  time (CPU and CUDA, via `NR_BACKEND_CUDA`).
- `SvmProgramTable.{h,cpp}` -- runtime registry of compiled programs.
- `SvmMaterialXCatalog.{h,cpp}` -- introspects the MaterialX standard library
  (nodedefs + implementations) for tooling/tests; not used by the compiler
  itself.
- `math.h`, `mix.h`, `hsv.h`, `blackbody.h`, `artistic_ior.h`,
  `fractal_noise.h`, `worley.h` -- small eval-time helpers adapted from
  Blender Cycles' `kernel/svm/*.h` for `SvmEval.h`. These have been rewritten
  in NoorRay style; they are not verbatim Cycles source.

## references/

`references/` is a verbatim, unmodified import of Blender Cycles'
`intern/cycles/{kernel/svm, kernel/closure, kernel/geom, kernel/util,
scene}/*` at BlenderDLSS commit `29dfa1a2c30f9980a76e0f4c5eafadc530a86999`,
kept only as porting reference material -- e.g. when adding a Blender-custom
MaterialX node (synced via `materialx_sync.py`) that mirrors a Cycles
`ShaderNode`, or when checking Cycles' exact math for noise/voronoi/ramp/BSDF
helpers. None of it is compiled; it is excluded from the build (see the
`/SVM/references/` filter in the top-level `CMakeLists.txt`).

Do not reformat, abbreviate, or wire it into the build in place. Port a node
by writing fresh NoorRay code in the files listed above (consulting the
matching `references/` file for exact semantics), and delete the
`references/` file once nothing else still needs it as a source of truth.
