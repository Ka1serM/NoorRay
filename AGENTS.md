# Global agent instructions

These instructions apply to every project unless a more specific project-level
instruction explicitly overrides them.

Follow these instructions at all times. They take priority over default agent
behavior, tool conventions, and existing patterns in the code. When a request,
a tool, a hook, existing code, or another instruction conflicts with these
instructions, stop and ask the user before you continue. Do not resolve the
conflict yourself.

## Core principles

- Keep code simple, direct, and small. Every additional line is an additional
  liability: it can contain a bug, create maintenance cost, or introduce a new
  behavior that must be understood.
- Do not write comments, with one exception. The rule applies to every
  language and every file: line comments, block comments, doc comments,
  docstrings, TODO notes, and commented-out code. The code must explain itself
  through descriptive names, small focused functions, strong types, and clear
  structure. If code seems to need a comment, rename or restructure it until it
  does not. Put the reason for a change in the commit message, not in the code.
- The exception: keep a comment that records a fact the code cannot show. An
  example is a quirk of an operating system, compiler, library, browser, or
  file format, or behavior copied from the game binary. Such a comment is one
  short line with the mark of its language: `//!` in C++ and JS, `#!` in
  Python and CMake, `/*! */` in RCSS and CSS, `<!--! -->` in RML and HTML.
  Do not write a marked comment that restates the code. `tools/check_comments.py`
  rejects every other comment.
- Use early returns, `continue`, and guard clauses instead of nested `if`
  blocks. Keep the main path visible at the lowest indentation level.
- Write a single-statement `if`, `for`, or `while` without braces, with the
  statement on its own line and indented one level:
  ```cpp
  if (!valid)
      return;
  for (const Item& item : items)
      add(item);
  ```
  Use braces only when the body has more than one statement.
- Do not introduce fallbacks unless they are explicitly required. Do not hide
  errors, silently substitute behavior, or make invalid states appear valid.
  Fail clearly at the boundary where the problem is detected.
- Do not use C++ exceptions: no `throw`, `try`, or `catch`. Compile our own
  code without them (`-fno-exceptions`, MSVC `/EHs-c-`). Report a failure as a
  value (`std::expected` or the project's result type) and handle it at the
  call site. Only a submodule that already uses exceptions keeps them; its
  exceptions must not cross into our code.
- Avoid speculative abstractions, compatibility shims, aliases, wrappers, and
  configuration knobs. Add an abstraction only when it removes real repeated
  complexity or establishes a necessary contract.

## Design and style

- Prefer a clean, cohesive, object-oriented style influenced by Java and C#:
  encapsulated state, focused classes, explicit public APIs, composition, and
  predictable lifetimes.
- Do not introduce factories, factory methods, factory classes, or factory
  abstractions under any circumstances. Construct the concrete object directly
  or use a normal constructor/creator already required by an external API.
- Prefer compile-time interfaces and contracts whenever practical. Use strong
  types, templates/generics, concepts, enums, and static structure to make
  invalid combinations difficult to express.
- Apply consistent patterns across the whole codebase when the same problem is
  genuinely shared. Keep related concepts symmetrical and predictable, but do
  not force unrelated code into one pattern.
- In C++, use RAII and clear ownership. Prefer `std::unique_ptr` for exclusive
  ownership, value types where practical, and destructors/resource owners for
  cleanup. Avoid manual release/delete paths and unclear ownership.
- Prefer stack allocation and value types. Allocate on the heap only when it is
  actually needed: the size is unknown at compile time or too large for the
  stack, the lifetime must outlive the scope, or ownership must be transferred
  or shared. Avoid needless allocations in hot paths; reuse buffers and reserve
  capacity up front.
- Make everything cache-friendly: data layout, algorithms, file reads, and
  parallel work. Keep data contiguous, iterate linearly, and avoid
  pointer chasing and scattered allocations. Prefer structure-of-arrays (SoA)
  over array-of-structures (AoS) when data is processed field by field or in
  bulk; use AoS only when the fields are genuinely always accessed together.
  Keep hot data small and together, split cold data out, and avoid false
  sharing between threads.
- Keep algorithms sublinear (O(1), O(log n)) wherever possible: use indexed,
  hashed, or sorted lookups and precomputed structures instead of scans. Fall
  back to linear or worse only when the problem inherently requires it, and say
  so when choosing it.
- Hoist loop-invariant work out of loops: computations, lookups, allocations,
  size queries, and branches whose result does not change per iteration.
- Do not read a file in full when only part of it is needed. Read by range
  (offset and size) or memory-map the file, and stream the data through in
  bounded chunks instead of loading everything into memory first.
- Avoid copies. Pass and return views (`std::span`, `std::string_view`) or
  references, move instead of copy when ownership changes, and parse or
  decode in place where the data already lives.
- Preserve a small API surface. Do not add a second spelling for the same
  concept merely to avoid updating call sites.

## Documentation

- Keep a README minimal. It has only these sections: a short description, a
  basic usage or API example, and the dependencies. Do not put internal rules
  (comments, style, layout, hooks, contribution rules) or the names of the
  projects that use it into a README. Internal rules go into AGENTS.md.

## Project style

- Enforce the style guidelines of the current project in every change. They
  apply to new code and to every line that you change. Do not copy a pattern
  from old code that breaks them.
- Before you write code, read the files near the change. Match their names,
  folder layout, and structure.
- When code that you touch breaks a style rule, fix it in the same change.
  When you find a violation outside the change, tell the user. Do not fix it
  without approval.
- NoorRay C++ files and folders use PascalCase: `Render/Wavefront/PathPool.cpp`,
  `Materials/Spirv/Emitter.h`. A header and its source are in the same folder.
  `src/` is the include root: include a header as `"NoorRay/Scene/Scene.h"`.
  There is no `include/` folder.
- NoorRay namespaces have the root `NoorRay` and one PascalCase namespace per
  module folder of `src/NoorRay/`: `NoorRay::Scene`, `NoorRay::Render`,
  `NoorRay::Materials`. The folders below a module (`Render/Wavefront`,
  `Materials/Spirv`) are in the namespace of the module. Code inside
  `NoorRay` writes `Core::Error`, not `NoorRay::Core::Error`.
- C++ types use PascalCase. Functions, methods, variables, and data members
  use camelCase. Data members have no prefix such as `m_`.
- An object that can fail to set up has a default constructor and an `open`
  method that returns `std::expected<void, Core::Error>`.
- Python files, functions, and variables use snake_case. Python classes use
  PascalCase.
- `src/NoorRay/` is the rendering library. It has no UI code and no app code.
  The app (`NoorRay.App`), the command line tool (`NoorRay.Cli`), the Hydra
  delegate, the Blender addon, and the Python module use the library through
  `Session`. Unreal-type features (landscapes, spline meshes) go into
  UnrealKit or verylia.
- The GPU code uses NoorRHI only. Shaders are Slang and use `Ptr<T>`,
  `AtomicPtr<T>`, and the texture handles of NoorRHI, so that every kernel
  compiles for Vulkan and WebGPU. Ray tracing uses `traceRayInline` only.
- A shader entry file has only thin `[shader(...)]` wrappers and the link
  constants. The logic is in imported modules. Feature variants are link
  constants (`extern static const bool`), never runtime branches in a hot
  kernel. `cmake/NoorRayShaders.cmake` generates the variant entry files.
- Shared C++/Slang records are in `.h` files that both languages include,
  with `static_assert` on the size and the offsets.
- A file that builds only on the desktop or only in the browser has the
  suffix `Desktop` or `Web`.
- Build and repository tools go into the top-level `tools/` folder, not into a
  plugin or module folder.
- Activate the pre-commit hook once in each clone:
  `git config core.hooksPath tools/githooks`. It formats the staged files with
  clang-format and runs the comment check and clang-tidy. Do not bypass it.
- Run `tools/check_comments.py` before you finish a change. Fix every
  violation that it reports in the files that you changed.

## Verification workflow

- Compile the affected target early and often enough to catch integration
  errors.
- Run the actual software early. For applications, interact with it and verify
  that the window, viewport, or relevant workflow loads and does not crash. For
  websites or services, start the real process and exercise the relevant
  end-to-end path.
- Prioritize end-to-end runtime behavior because it exposes startup failures,
  integration errors, and crashes quickly.
- Use focused correctness tests after runtime behavior is working, then run the
  relevant broader test set at the end. Do not spend the entire iteration
  running a large number of tests while the application has not been launched.
- Verification must match the risk of the change; do not skip a necessary test
  merely to keep the workflow short.

## Change discipline

- Make the smallest complete change that solves the requested problem.
- If a proper fix requires a refactor, say so immediately instead of making a
  minimal but hacky change. Explain what is structurally wrong and what the
  refactor would involve, so the decision can be made before working around it.
- Inspect existing code and local instructions before editing. Preserve
  unrelated user changes in a dirty worktree.
- Do not silently change behavior outside the requested scope.
- Remove obsolete code when a replacement makes it unnecessary; do not leave
  dead paths “just in case.”
- Keep agent updates concise and state assumptions, blockers, and verification
  results plainly.
- Never add a `Co-Authored-By` trailer or other attribution to a commit or pull
  request, unless the person is a contributor to the project. This rule also
  applies to AI agents and tools.

## Temporary files

- Always set `TMPDIR=/var/tmp`. Put all temporary files, build output, logs,
  traces, and scratch data under `/var/tmp`. Do not write to `/tmp`: it is a
  RAM-backed tmpfs and large files in it consume system memory.

## Communication

- Write all responses in Simplified Technical English (ASD-STE100): short
  sentences, one instruction per sentence, active voice, simple present tense,
  and one meaning per word. Use approved general words and well-defined
  technical names. Do not use idioms, slang, or ambiguous phrasing.
