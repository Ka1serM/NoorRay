#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
build_dir="${1:-${repo_root}/build/energy-luts}"

cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target NoorRayEnergyLuts -j"$(nproc)"

echo "Energy LUT headers regenerated in ${script_dir}/Generated"
