# Scenarios and Methods

DiffRenderer compares differentiable renderers **against each other on the same task**.

- **Scenario:** a task, for example "reconstruct a scene from photographs". It fixes:
  - the dataset (train and test views)
  - the loss
  - the budget (iterations and wall-clock time)
  - the metrics
- **Method:** a representation together with a renderer, for example "3D Gaussians + splatting" or "mesh + texture + path tracing". A method lists the scenario kinds it can enter.

The toy problems in `Algorithms/` stay as they are. They are the edge and discontinuity study (E1): each one has a known true parameter set.

## Scenario kinds

| Kind | Datasets | Loss | Metrics |
|---|---|---|---|
| `Scene` | Mip-NeRF 360: garden, bonsai, counter | 0.8 L1 + 0.2 D-SSIM | test PSNR, SSIM, LPIPS (offline), train time, render ms, primitives, memory |
| `Object` | NeRF-Synthetic: the 8 objects | same as `Scene` | same as `Scene`, plus mask IoU on test views |
| `Volume` | MSD BrainTumour (FLAIR), 96³ head phantom | L2 | test PSNR/SSIM, transfer-function error where the method has one, re-colouring test |

### Conventions

These follow the published papers, so that our numbers can be compared with theirs.

- **NeRF-Synthetic:**
  - 800×800 images.
  - RGBA is composited on a **white** background.
  - Splits: `transforms_train.json` (100 views) and `transforms_test.json` (200 views).
  - The camera is OpenGL style (x right, y up, looking along −z). It is converted to OpenCV style (x right, y down, z forward).
  - The 3DGS start: 100k random points in [−1.3, 1.3]³ with random colours.
- **Mip-NeRF 360:**
  - Image folders: `images_4` for the outdoor scenes (garden, bicycle, stump, flowers, treehill) and `images_2` for the indoor scenes (bonsai, counter, kitchen, room).
  - Every 8th image (sorted by name, index % 8 == 0) is a test view. The rest are train views.
  - Cameras come from `sparse/0/cameras.bin` and `images.bin` (COLMAP, PINHOLE or SIMPLE_PINHOLE, rescaled to the image folder).
  - The start point cloud comes from `points3D.bin`.
  - Scene extent: 1.1 × the largest camera distance from the mean camera centre, as in 3DGS.
- **Volume:**
  - The volume is normalized to [0, 1] and placed in the unit cube [−0.5, 0.5]³.
  - A **reference transfer function** colours it. Targets are rendered with a reference emission-absorption ray marcher with a fine step.
  - Views: 64 train views and 32 test views on a sphere, from a Fibonacci spiral. Test views lie between the train views.
  - Two resolutions: 256² for the phantom and 512² for MRI.

## Published reference numbers (validation)

Each method must come close to its paper on at least one dataset before we use its numbers. The test PSNR (dB) to check against:

| Method | NeRF-Synthetic mean | Mip-NeRF 360 garden | Mip-NeRF 360 bonsai | Source |
|---|---|---|---|---|
| 3DGS (30k it) | 33.31 | 27.41 | 31.98 | Kerbl et al. 2023, Tab. 1, 6 |
| 2DGS (30k it) | 33.07 (approx.) | — | — | Huang et al. 2024 |
| Plenoxels | 31.71 | — | — | Fridovich-Keil et al. 2022 |

Check the exact values in the papers when a method is finished. Our ports may differ by a few tenths of a dB because of the random start and the float summation order.

## Methods

| Id | Representation | Renderer | Kinds | Notes |
|---|---|---|---|---|
| `3dgs` | 3D Gaussians, SH degree 3 | tile-based splatting | Scene, Object, Volume | densification, opacity reset, as in the paper |
| `3dgrt` | 3D Gaussians, SH degree 3 | ray tracing of proxy meshes, sorted by peak | Scene, Object, Volume | the 3DGS training; composites linear radiance, as the spectral path tracer does |
| `2dgs` | 2D Gaussian surfels | ray–splat intersection | Scene, Object, Volume | depth distortion and normal consistency terms |
| `grid` | dense or sparse voxel grid: density + SH | emission–absorption ray marching | Scene, Object, Volume | Plenoxels-like, coarse-to-fine, TV regularizer |
| `sdf` | SDF + colour voxel grid | sphere tracing of the opaque surface, relaxed boundary term (Wang et al. 2024) | Scene, Object | coarse-to-fine from 32³, redistancing after every step, mesh by marching cubes |
| `mesh` | triangle mesh + PBR textures (kd, ks, normal) + learned environment light (nvdiffrec, Munkberg et al. 2022) | differentiable rasterizer (analytic antialiasing), split-sum image based lighting | Object (Scene: attempt) | pass 1: vertex materials, continuous remeshing with large steps instead of DMTet; pass 2: fixed topology, xatlas textures; see the deviations below |
| `mesh-pt` | triangle mesh + texture + light | path tracer with edge sampling | Object | slow, few views per step |
| `dvr-tf` | the known volume + a 1D RGBA transfer function | ray marching | Volume | only the transfer function is learned |

### Deviations of `mesh` from nvdiffrec

`mesh` follows the reference code of nvdiffrec for the materials, the shading, the light and the regularizers. It deviates in these points:

- **Geometry:** pass 1 moves a remeshed surface with large steps (Nicolet et al. 2021). It does not use DMTet. Thus pass 1 has a kd and ks on each vertex, not an MLP texture.
- **Loss:** the shared loss on sRGB images. nvdiffrec uses a log L1 loss on tone mapped linear images.
- **Background:** the background of the dataset. nvdiffrec composites the targets on a random background.
- **Gradients:** the image gradient reaches the positions through the vertex normals. It does not reach them through the tangents, the face normals or the view vectors.
- **kd smoothness:** only in pass 2, because pass 1 has no texture to jitter.
- **Batch:** one view for each step. nvdiffrec uses 8.
- **Light in pass 2:** learned, as the default of nvdiffrec. The hotdog and materials configurations of nvdiffrec lock it.

## Interfaces (C++)

- **`Scenes::Dataset`:** the train and test `View`s (camera + image + optional mask), a start point cloud, the background, the bounds, and an optional `Scenes::Volume` (the known scalar volume, for `dvr-tf`).
- **`Methods::Method`:** `open(device, dataset, settings)`, `step(target)` for one iteration on one train view, `render(camera, image)`, and `getStatistics()`.
- **`Training::Loss`:** the shared GPU loss, 0.8 L1 + 0.2 D-SSIM or L2. It writes the image gradient to a GPU buffer. Every method records it between its forward and backward pass. The loss is the same for every method.
- **`Training::Comparison`:** trains every chosen method on one scenario with the same budget. It evaluates every N iterations and at the end on all test views, and writes one report.
- **LPIPS:** the app writes the test renders as PNG. `tools/lpips.py` (PyTorch `lpips`, VGG) adds LPIPS to the report.
