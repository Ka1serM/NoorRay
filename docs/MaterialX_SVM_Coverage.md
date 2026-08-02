# MaterialX SVM coverage and closure report

## Executive status

The current implementation is a fixed bytecode SVM with a MaterialX graph
compiler in front of it. It is not yet complete for every non-volume
MaterialX node, but the runtime boundary is MaterialX-only.

The architecture is now on the correct track:

- one fixed interpreter and one compact bytecode stream;
- NaN-encoded literal/stack operands;
- a 255-float stack with first-fit lifetime reuse;
- append-only opcode numbering for bytecode compatibility;
- fixed-size closure storage with explicit overflow failure;
- MaterialX nodegraphs flattened before native SVM compilation.

The remaining work is semantic coverage, not a change to the overall
architecture.

## Fixed-stack execution model

The implementation now follows a fixed-stack execution contract:

- MaterialX is the front-end graph language; compilation emits a flat,
  append-only `uint32` node stream with POD payloads;
- every value input is a literal or NaN-encoded fixed-stack reference;
- the runtime is one iterative interpreter loop over that stream, with a local
  255-float stack and no runtime graph traversal or dynamic allocation;
- closure nodes write into a fixed ShaderData-style closure pool, then the pool
  computes selection probabilities and the composite MIS PDF before sampling;
- closure mix factors are clamped and compiled as conditional stream branches,
  using `JUMP_IF_ZERO`/`JUMP_IF_ONE` stream branches;
- stack slots are assigned and reclaimed from last-use information during
  compilation.

MaterialX-specific substitutions are intentionally limited to the front end,
node payloads, and the supported closure payload table. Unsupported node and
closure types do not alter this execution model; they fail compilation with a
named MaterialX category.

The vendored MaterialX library contains 802 unique `ND_` definitions across the standard, PBR, color-management, NPR, lights and BXDF libraries (excluding target implementation duplicates). Those definitions cover 218 node categories. The compiler's explicit native/terminal dispatch currently names 114 categories; additional categories are eligible for nodegraph flattening, while the remainder require new semantics or an explicit unsupported status.

## Implemented SVM foundation

The current native compiler/evaluator handles the following families:

- scalar math, logical operations, comparisons, remap/range, clamp, gamma, HSV, contrast, saturation and color conversion;
- vector math, dot/cross/distance/normalize/reflect/refract;
- combine/separate channels, including color4/vector4 combine, separate, premultiply and unpremultiply;
- scalar unary/binary arithmetic, comparisons, mix, clamp and range across
  vector2/vector3/vector4 values using shared component instructions;
- UV, position, normal, tangent, bitangent, vertex color, view direction and known MaterialX geometry properties;
- mapping, rotate2d/rotate3d, gradients, time/frame, typed matrix values and
  point/vector/normal transforms, matrix arithmetic, transpose, determinant,
  inverse, and matrix conditionals;
- image sampling, MaterialX-standard normal maps, checker, Perlin/fractal,
  Worley/cell/unified noise and blackbody;
- standard geometric nodes, object/world point-vector-normal transforms,
  `viewdirection`, and `noise3d` white-noise export targets;
- MaterialX diffuse, conductor, dielectric, sheen, subsurface and uniform EDF leaves;
- metallic BSDF lowering to the conductor closure path using explicit complex
  IOR and extinction inputs;
- OpenPBR and standard_surface terminals;
- surface/add/mix/multiply/layer closure composition and opacity output.

MaterialX standard nodegraphs such as tiled images, ramps, switches and several conversion/PBR wrapper graphs are lowered through MaterialX graph flattening when their resulting primitives are supported.

The SVM mix instruction is intentionally limited to the MaterialX compositing
categories it receives (`mix`, `plus`, `minus`, `difference`, `screen`,
`overlay`, `dodge`, and `burn`). Blender-only blend policy is expanded by the
upstream exporter into standard MaterialX arithmetic, conditional, and HSV
nodes before SVM compilation.

## Explicitly outside the requested scope

The following MaterialX families are intentionally excluded from the
non-volume surface target requested for NoorRay:

- `absorption_vdf`, `anisotropic_vdf`, `layer_vdf`, `mix_vdf`,
  `multiply_vdfC/F`, `add_vdf`, `volume`, and `volumematerial`;
- `directional_light`, `point_light`, `spot_light`, `light`, and the other
  light-shader nodes;
- `uniform_edf`, `conical_edf`, `generalized_schlick_edf`, and
  `measured_edf` as an EDF family. `uniform_edf` remains implemented because
  it is also used by the unlit terminal.

These are catalogued and named in the UI, but are not completion blockers for
the requested SVM surface scope.

## Known incomplete or approximate non-excluded coverage

These are not “done” and must not be counted as 1:1 parity:

- `bump_vector3` and `heighttonormal_vector3`: exact derivatives/ray
  differentials are not available in the current shading context;
- arbitrary `geompropvalue_*` and `geompropvalueuniform_*`: only the
  renderer's known built-in attributes are mapped; arbitrary named primvars
  and interpolation metadata are not;
- `blur_*`, `hextiledimage_*`, `hextilednormalmap_vector3`, and the remaining
  texture-space wrapper graphs requiring derivatives or specialized filtering;
- `chiang_hair_absorption_from_color`, `chiang_hair_bsdf`,
  `chiang_hair_roughness`, and `deon_hair_absorption_from_melanin`;
- `generalized_schlick_bsdf` (exponent and anisotropic roughness are not
  preserved) and `layer_bsdf` (no exact directional top-layer attenuation);
- metallic/conductor anisotropy, rotation, tangent-oriented GGX, and thin-film
  controls are not yet consumed by the conductor lobe;
- Object Info and Camera Data are approximated by standard geometric,
  arithmetic, and `randomfloat` nodegraphs; no private geomprops are used;
- camera/custom Vector Transform spaces are approximated as world space;
- Blender White Noise is intentionally mapped to standard MaterialX `noise3d`,
  so its noise implementation is not asserted to be bit-identical to
  Blender's;
- `displacement_float`, `displacement_vector3`, and displacement-shader
  composition nodes;
- compound color-processing paths such as `colorcorrect_*` and other
  specialized color-management/filter nodes;
- the specialized closure families `Lama*` and the remaining measured,
  hair, or host-integration nodes; their nodegraph wrappers are catalogued,
  but they are not silently lowered to a different closure.

Many other named definitions (`UsdPreviewSurface`, glTF wrappers, ramps,
switches, color transforms, and procedural nodegraphs) are flattened through
MaterialX's own nodegraph implementations when their primitive dependencies
are supported. A catalog entry therefore does not by itself mean that every
runtime signature is exact.

The compiler should continue to reject unsupported surface nodes rather than silently lower them to an incorrect closure. The intentional volume exclusion is not evidence of complete non-volume coverage.

## Closure system

The closure system now has the right fixed-pool behavior for an SVM backend:

- up to 64 active lobes, matching Blender's current `MAX_CLOSURE` scale;
- negative weights are clamped to zero;
- weights below `1e-5` are discarded;
- closure-pool overflow is recorded and causes SVM evaluation failure at `End`;
- mixture PDFs are recomputed over the full composite for non-singular samples;
- delta/singular samples retain their selected-lobe probability treatment;
- nested MaterialX add/mix/multiply trees propagate RGB/spectral weights.

It is not yet a generic MaterialX closure system in the strongest sense. The storage union currently has only diffuse, conductor and dielectric payloads. Sheen and subsurface are mapped onto those existing payloads, and EDFs are evaluated separately. There is no generic closure type registry, no arbitrary closure payload allocation, and no exact layer/coat attenuation model yet.

## Verification

The focused build and tests currently pass:

- `svm_materialx_compiler_test`: 16 test cases, 29 assertions;
- `composite_bsdf_test`: 2 test cases, 71 assertions;
- `materialx_node_catalog_test`: 4 test cases, 3225 assertions;
- `dielectric_bsdf_test`: 8 test cases, 330066 assertions;
- `opaque_bsdf_test`: 9 test cases, 2091 assertions;
- `diffuse_lobe_test`: 2 test cases, 11 assertions;
- `git diff --check` passes.

These tests verify the bytecode foundation, representative MaterialX compilation and closure regressions. They do not yet constitute exhaustive node-by-node MaterialX parity tests.

The SVM/device library targets build successfully. A workspace-wide build is
currently blocked by unrelated existing targets: `RayTest` and
`LightSamplingTest` use older Ray/light APIs, and
`materialx_graph_node_registry_test` has an unresolved `exposedInputs` link
symbol. Those failures are outside the SVM implementation.

## Required completion gates for “1:1 MaterialX SVM”

The implementation should only be called complete after it has:

1. a generated manifest for every MaterialX nodedef in the vendored library;
2. a test material for every supported non-volume nodedef and every output type;
3. explicit status per node: native, flattened, exact, approximate or unsupported;
4. typed vector2/vector3/vector4 and matrix stack/value handling;
5. renderer derivatives and exact bump/normal semantics;
6. generic primvar lookup and interpolation rules;
7. native hair and exact layer closures for the non-excluded surface scope;
8. closure-pool, mixture-PDF, energy and bytecode parity tests against reference evaluations.

Until those gates are met, the honest status is “MaterialX SVM foundation
with substantial coverage,” not “complete 1:1 MaterialX SVM.”
