# Blender to MaterialX exporter coverage

## Current status

The exporter is not yet a complete Blender shader-node translation. It is a
good surface-material bridge for the currently supported SVM subset, but it
must not be counted as complete merely because it produces valid MaterialX.

The important correction in this pass is that unsupported nodes are no longer
invisible to callers. `export_material()` now returns
`ExportedMaterial.unsupported_nodes`, the Blender validation script fails when
one is reached, and the live sync layer prints the exact Blender node types
that were defaulted.

The audit was performed against Blender 5.2.0 LTS:

- 100 registered `ShaderNode*` classes were enumerated;
- the exporter has 48 declared `_export_*` methods, plus aliases for
  anisotropic/velvet and group/reroute plumbing;
- the missing-class count includes output nodes and the intentionally excluded
  volume/light families, so it is not itself a percentage of usable surface
  coverage.

## Covered exporter families

- Principled BSDF to `open_pbr_surface`;
- diffuse, translucent, transparent, glass, refraction, glossy/anisotropic,
  sheen/velvet, subsurface and emission closures;
- Mix Shader/Add Shader closure composition;
- nested node groups, group inputs/outputs and reroutes;
- value/RGB/blackbody constants;
- scalar and vector math, mapping, combine/separate, mix, clamp, map range,
  gamma, hue/saturation, invert, brightness/contrast, ramps and sampled RGB
  curves;
- all 19 Blender Mix/MixRGB blend modes: Mix, Darken, Multiply, Burn, Lighten,
  Screen, Dodge, Add, Overlay, Soft Light, Linear Light, Difference,
  Exclusion, Subtract, Divide, Hue, Saturation, Color, and Value;
- UV/texture coordinates, image textures, checker, gradient, wave, noise and
  Voronoi textures;
- normal maps and bump nodes (the latter remains derivative-limited in the
  SVM runtime);
- Fresnel, layer weight, vertex colors, named Attribute nodes through standard
  `geompropvalue`, and known geometry values;
- Vector Transform (object/world spaces), White Noise, and Metallic BSDF (via
  the standard `generalized_schlick_bsdf` closure). Camera View Vector maps to
  `viewdirection`;
- New Geometry random outputs and Light Path outputs through standard
  MaterialX spatial/arithmetic approximations.

The exporter emits MaterialX node categories, not Blender-renamed equivalents.
The downstream SVM compiler remains the authority on whether an emitted node
is native, flattened, approximate, or unsupported.

## Critical cross-layer gaps

These are more important than the raw handler count because the exporter can
emit a node that the SVM accepts while dropping part of its Blender meaning:

- Principled/OpenPBR fields such as anisotropic roughness, subsurface radius
  and anisotropy, thin-film controls, and thin-wall behavior are emitted by
  the exporter but are not all represented in `NodeClosureOpenPbrSurface` or
  consumed by the SVM evaluator yet.
- Glossy/anisotropic uses an artistic-IOR approximation rather than Blender's
  exact distribution and multiscatter behavior.
- Subsurface is emitted as a generic MaterialX subsurface closure, while the
  current NoorRay closure payload is still an approximation.
- Bump is emitted as `bump`, but exact MaterialX derivative semantics require
  renderer ray differentials that are not currently in the shading context.
- Blender nodes returning RGBA from image textures now retain the alpha slot
  as a true fourth SVM component; this path must remain covered by runtime
  tests because alpha is also exposed through MaterialX's named `extract`
  output.

## Blender surface nodes still needing handlers

These are the high-priority non-volume/non-light gaps from the Blender 5.2
inventory:

- `ShaderNodeAmbientOcclusion`, `ShaderNodeBevel`,
  `ShaderNodeNormal`, `ShaderNodeTangent`, `ShaderNodeParticleInfo`,
  `ShaderNodePointInfo`,
  `ShaderNodeRaycast`, `ShaderNodeWireframe`;
- `ShaderNodeBackground`, `ShaderNodeBsdfHair`,
  `ShaderNodeBsdfHairPrincipled`,
  `ShaderNodeBsdfRayPortal`, `ShaderNodeBsdfToon`, `ShaderNodeDisplacement`,
  `ShaderNodeEeveeSpecular`, `ShaderNodeHoldout`;
- `ShaderNodeFloatCurve`,
  `ShaderNodeSqueeze`, `ShaderNodeVectorCurve`, `ShaderNodeVectorDisplacement`,
  `ShaderNodeVectorRotate`, `ShaderNodeWavelength`;
- `ShaderNodeLightFalloff`, `ShaderNodeHairInfo`,
  `ShaderNodeRadialTiling`, `ShaderNodeShaderToRGB`, `ShaderNodeUVAlongStroke`;
- `ShaderNodeTexBrick`, `ShaderNodeTexEnvironment`, `ShaderNodeTexGabor`,
  `ShaderNodeTexMagic`, and `ShaderNodeTexSky`.

Several of these remain difficult to approximate without renderer data:
- arbitrary custom geometry primvars (the exporter preserves their names and
  defaults, but the current NoorRay mesh payload only provides the standard
  built-in streams), particle data, raycast, pointiness, Eevee-only
Shader-to-RGB, hair closures, and exact displacement.

The newly covered mappings have these runtime limits:

- Object Info is approximated with standard `position`, `geomcolor`,
  `extract`, `magnitude`, and `randomfloat` nodes. Location is surface world
  position, Color is vertex color, indices are approximate/randomized values,
  and Material Index is zero.
- Camera View Vector maps to standard `viewdirection`. View Distance is the
  magnitude of world position and View Z Depth is a position/view-direction
  dot product; both are deliberate standard-node approximations.
- Vector Transform supports point, vector, and normal conversions between
  object and world spaces. Camera/custom spaces are approximated as world.
- White Noise is exported as the standard MaterialX `noise3d` node with
  standard vector arithmetic; its noise implementation is intentionally not
  expected to be bit-identical to Blender's.
- New Geometry Random and Random Per Island use standard `position`,
  `magnitude`, and `randomfloat`; MaterialX has no mesh-island identifier, so
  this is spatial rather than island-stable.
- Light Path uses standard primary-camera constants (`Is Camera Ray` and
  `Is Diffuse Ray` true, other ray-kind flags false), `magnitude(position)` for
  Ray Length, and zero for bounce/depth outputs. Default MaterialX exposes no
  ray-type or bounce-query primitive.
- Metallic F82 mode maps Blender Base Color/Edge Tint and roughness to the
  standard `generalized_schlick_bsdf` closure; Physical mode maps IOR/Extinction
  to `conductor_bsdf`. Anisotropy, rotation, and thin-film controls are not yet
  preserved by the current SVM closure payload.

## Intentionally excluded or non-surface

- `ShaderNodeVolumeAbsorption`, `ShaderNodeVolumeCoefficients`,
  `ShaderNodeVolumeInfo`, `ShaderNodeVolumePrincipled`, and
  `ShaderNodeVolumeScatter`;
- `ShaderNodeOutputLight`, plus light-only texture/output paths such as IES;
- `ShaderNodeOutputMaterial`, `ShaderNodeOutputWorld`, `ShaderNodeOutputAOV`,
  `ShaderNodeOutputLineStyle`, and the node-tree/output plumbing classes.

## Required exporter completion gate

After the MaterialX/SVM layer reaches the planned ~97% surface coverage, the
next phase should be driven by a Blender-generated matrix of every reachable
node/socket/output combination:

1. instantiate each supported Blender node in Blender;
2. export every output socket, including alternate outputs and RGBA alpha;
3. validate the resulting MaterialX document;
4. compile it through SVM with the corresponding MaterialX nodedef;
5. compare Blender reference values against SVM values for deterministic value
   nodes and compare rendered outputs for closures/textures;
6. fail the test if `unsupported_nodes` is non-empty;
7. record each result as exact, flattened, approximate, excluded, or missing.

## Verification

The Blender exporter validation passes on Blender 5.2.0 LTS. It now includes
a 76-case Mix/MixRGB mode matrix (19 modes across each supported Mix data type
and MixRGB), validates every generated MaterialX document, and asserts that HSV
and Soft Light modes use standard MaterialX node expansions. The focused SVM
and MaterialX catalog tests remain the authoritative downstream checks.
