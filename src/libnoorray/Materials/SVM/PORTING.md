# MaterialX SVM

This directory contains NoorRay's MaterialX-to-bytecode compiler and its
CPU/CUDA interpreter.

- `SvmMaterialXCompiler.cpp` walks a MaterialX document and emits bytecode.
- `SvmTypes.h` defines the stable instruction and payload formats.
- `SvmEval.h` evaluates those instructions at shading time.
- `SvmMaterialXCatalog.{h,cpp}` exposes the standard-library node catalog to
  tooling and tests.
- The helper headers implement MaterialX standard-library math, color,
  procedural, and closure support for the interpreter.

The SVM accepts MaterialX semantics only. Application-specific node names,
socket conventions, and translation policy belong in the upstream exporter
and must not be added to this directory.

When adding support, start from the MaterialX nodedef and implementation in
`external/materialx/libraries`, then add the smallest matching instruction or
instruction sequence. Keep the compiler and interpreter in lockstep, and add
an XML/compiler/evaluation regression test for every new category.
