# Blender animation plan

Goal: render the 3D modelling final animation (`FinalHQ.blend`, Blender 5.2, Cycles) in NoorRay with high accuracy.

## Scope

In scope:

1. Blender exporter that writes NoorRay scene files (`.nrs`) with animation.
2. Animation: transforms, visibility, cameras, lights and deforming meshes for each frame.
3. Command line rendering of one frame or a frame range.
4. Material node graphs, including light shaders (gobos, noise cookies).
5. Light linking.
6. Blender-exact lights.
7. Procedural volume density and the Draine phase function.

Out of scope for now: adaptive sampling, bounce limits per lobe type, pixel filters, motion blur, a new display
transform, the compositor, and a native `.blend` reader.

## Decisions

| Topic | Decision |
|---|---|
| Reading `.blend` | Blender reads it. The exporter runs in Blender and takes the evaluated depsgraph, so modifiers, Geometry Nodes, constraints, drivers and Alembic caches need no NoorRay code. |
| Exporter | `noorray.export_scene` of the addon module. It steps through the frames, syncs each one with `SceneSync` (the same code that the render engine uses), and gives the live scene to `Format::AnimationRecorder`. The addon operator and `export_scene.py` both call it. |
| Coordinates | Blender is Z-up, NoorRay is Y-up. Root transforms get a rotation of -90 degrees about X. Mesh data stays in object space. |
| Hierarchy | No nodes. Each object has its world transform. |
| Animation storage | Tracks of named property channels: a track targets one object of the scene (scene, instance, light, camera, material, environment, volume), and each channel keys one property (the names are in `Format/AnimationData.h`). The recorder samples every frame and removes the keys that interpolation reproduces. A change that no channel expresses is a key of the whole object. |
| Deforming meshes | Mesh sequences, only for meshes that change. A frame shows the last key frame at or before it. A full mesh is stored where the topology, the UVs, the colours, the attributes or the subsets change. Other key frames hold only positions when the mesh has a `MeshTopology` that rebuilds normals and tangents, and positions, normals and tangents without one. Stored corner normals stay in each key frame. |
| Textures | The scene holds the texels of each texture once. A texture that the sync reads again stays one texture. |
| Materials | `SceneSync` compiles the node trees. A node that it does not support is an export error, or a warning with `--approximate`. |

## Phases

### 1. Animated scene format and runtime (done)

- `Format::AnimationData`: the frame range, the frames per second, tracks, object keys and mesh sequences. The header
  lists every property. Transforms are the 12 floats of `Core::Transform`, and the program and graph parameter values of
  a material, light, environment or volume are properties too. `hidden` and `active_camera` are constant between keys.
  `Format::SceneWriter::write(scene, animation, path)` writes it.
- `Format::SceneFile::load` returns the `Format::Animation`. `Animation::apply` binary-searches each channel, skips
  tracks whose values did not change, and writes through the `Scene` setters, so `SceneChange` and `GpuScene` see
  only the changes.
- `Scene::setMesh`, `Change::MeshReplaced`: replaced meshes upload again; same-topology meshes upload all vertex
  streams and refit. An empty mesh drops its BLAS.
- `Session::setFrame(frame)`. `Session::importFile` applies the first frame.

### 2. Command line and exporter (done)

- `noorray <scene.nrs> <out_####.exr> [samples [width height]] --frames A-B` (or `--frames N`).
- `blender -b file.blend --python build/NoorRay.Blender/export_scene.py -- out.nrs [--frames A-B] [--approximate]`.
  The `NoorRay.Blender` target stages the script next to the addon folder, from which it imports the module.
- The operator `noorray.export_scene` writes the current frame or the frame range of the scene.
- `noorray.export_scene(depsgraph, path, *, approximate=False, frames=None)`: `frames` is None for a still scene, or the
  first and the last frame. The export restores the frame of Blender.

#### Frame steps

- `SceneSync` reads again only what `depsgraph.updates` names. Blender clears these updates after `frame_set`, so the
  sync runs in a `frame_change_post` handler, which Blender calls after it evaluates the frame and before it clears them.
- Blender switches to the camera of a timeline marker after the handlers. When the camera changed, the export sets the
  frame again. It also sets the first frame twice, because a first sync learns the instance attributes while it reads.
- Data that another `frame_change_post` handler changes is in the next frame, because Blender evaluates it later.

#### Recorder

- The output scene holds every object of every frame with its first value. Each live object has one output object. An
  object that goes away frees its output object; a new object takes a free one with the same name (an instance: the same
  random id, which comes from the object name and the persistent ids).
- An instance is hidden, and a light or volume is dark, at the frames where it is missing.
- An instance that comes back with a new mesh continues its old mesh as a mesh sequence. When other instances still show
  the old mesh, the instance becomes a second output instance.
- A channel starts at the first change of a property. Keys come only with changes, with one hold key before each change.
- A mesh is compared with its last stored streams only where `SceneChange` names it, or where meshes came or went. There
  is no copy of an unchanged mesh but the one in the output scene.
- A cut to another Blender camera becomes a second camera and an `active_camera` key.
- Keyed objects share their node tree with the first frame: an animated socket value changes the compiled parameter, not
  the stored tree.

### 2b. App playback (done)

- Opening an animated `.nrs` in the app sets the timeline range and frame rate from the file and looks through the
  active camera instead of framing the scene. Scrubbing and playback apply the frame, including camera switches; the
  view follows the active camera.

### 2c. Material displacement (done)

Geometry Nodes that only move vertices with a formula (the wave of `Ground`, which was 7.7 GB of positions frames)
become a material displacement that NoorRay evaluates on the GPU every frame.

- `Material.program` (`Materials::Program`): the displacement graph
  of the Material Output as a flat register program (see [Material programs](#material-programs)).
- Supported nodes: Value, Math, Vector Math, Wave Texture (Blender's, with its noise distortion), Combine/Separate
  XYZ, Attribute, Texture Coordinate (Generated, Object, Normal), Geometry (Position, Normal), Displacement, and
  Vector Displacement in object space. Anything else fails the export, or with `--approximate` warns and drops the
  displacement. `Displacement and Bump` displaces without the bump.
- Displaced meshes have `Mesh.topology` and the point attributes that the program reads (`Mesh.attributes`).
- `Gpu::MeshDisplacement` runs after `MeshBuffers::update`: four kernels (`Shaders/Entries/Displace*.slang`) displace
  the source vertices from their rest positions and normals, recompute Blender's face or angle-weighted vertex
  normals, and write the welded positions, normals and tangents. The BLAS refits. Generated coordinates use the rest
  bounds, as Blender does.
- Animated socket values (drivers and F-curves on `nodes["X"].inputs[i].default_value`) become program parameters;
  their channels are named `parameter:<i>`. A parameter change uploads only the registers.

#### Material programs

Registers are float4; a scalar fills all four lanes. `registers` holds every start value (constants and current
parameter values), at most 64. An instruction is two words: op, mode, output register, then four input registers.
`mode` is the `MathOperation` of Math (plus 128 to clamp), the `VectorMathOperation` of Vector Math, the component of
Separate, the index into `attributes` of Attribute, and `type + 2 * direction + 8 * profile` of Wave (Blender's enum
orders). `displacement` names the register with the object-space offset.

#### Re-authoring `FinalHQ.blend`

`FinalHQ_NoorRay.blend` is a copy in which `Ground` is static: its Geometry Nodes keep the grid, extrusion and
`noorray_base` (the position before the wave) and mute both Displace groups, and the material `Green NoorRay` adds
the same wave as material displacement, driven by the frame as before.

### 3. Material IR and light shaders

- `docs/Plan.md` section 4, limited to the nodes of the scene: Image Texture, Mapping, Tex Coord, UV Map, Attribute,
  Color Ramp, RGB Curves, Hue/Saturation, Gamma, Bright/Contrast, Invert, RGB to BW, Mix, Math, Vector Math, Clamp,
  Map Range, Separate/Combine XYZ, Noise (exact port of Blender's), Gradient, Wave, Normal Map, Bump, Geometry,
  Layer Weight, Fresnel, Camera Data.
- Closures: Principled, Glossy (anisotropic), Glass, Emission, one Mix Shader.
- Light shaders compile from the same IR. Next event estimation and emitter hits evaluate them. Spot lights give
  Blender's projected coordinates for gobo images.
- Animated material and light shader values go into the parameter block and become channels of material and light tracks.

### 4. Lights and volume

- Blender-exact spot cone and blend, radius with soft falloff, normalize, exposure, disk and rect area lights with
  spread.
- Light linking: link-set id per instance and volume, receiver mask per light, one alias table per link set, mask
  test on emitter hits and in MIS.
- Per-light camera and volume-scatter visibility.
- Procedural volume density from the IR, with the color ramp maximum as the majorant. Draine phase function.
