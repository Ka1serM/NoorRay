# NoorRay energy LUTs

These lookup tables are generated from NoorRay's own GGX visible-normal
sampler, correlated Smith masking, exact dielectric Fresnel, and rough-glass
reflection/refraction conventions.

Regenerate the checked-in header fragments with:

```sh
src/libnoorray/Materials/Shading/EnergyLut/generate_energy_luts.sh
```

An optional first argument selects the CMake build directory. The generator is
deterministic. For quick development checks, invoke `NoorRayEnergyLutGenerator`
directly with lower `--directional-samples` and `--average-samples`; committed
tables should use the defaults documented by `--help`.

The files under `Generated/` contain comma-separated normalized `uint16_t`
values. native uploads use filtered normalized images; CPU tests
use matching manual interpolation over the same arrays.
