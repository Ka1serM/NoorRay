#!/usr/bin/env bash
#! Builds NoorRay.Python with the linux-portable-release preset in a glibc 2.28 container, so the module loads in every Blender from 4.2 on.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
build="$repo/build-linux-portable-release"
image=noorray-portable-linux
engine=$(command -v podman || command -v docker)
sdk=$(readlink -f "${VULKAN_SDK:?VULKAN_SDK must point to the Vulkan SDK with slangc}")

"$engine" build --quiet -t "$image" -f "$repo/tools/portable_linux.Containerfile" "$repo/tools" > /dev/null
mkdir -p "$build"
#! The slangc of the Vulkan SDK needs a newer glibc than the container has, so it runs on the libraries of the host through the loader of the host.
cat > "$build/slangc" <<SCRIPT
#!/bin/sh
exec /host/lib64/ld-linux-x86-64.so.2 --library-path "$sdk/lib:/host/lib64" "$sdk/bin/slangc" "\$@"
SCRIPT
chmod +x "$build/slangc"

"$engine" run --rm --security-opt label=disable --userns=keep-id \
    -v "$repo:$repo" -v "$sdk:$sdk:ro" -v /usr/lib64:/host/lib64:ro -e "VULKAN_SDK=$sdk" -w "$repo" "$image" \
    bash -c "cmake --preset linux-portable-release -DNOORRHI_SLANGC=$build/slangc -DOPENDLSS_RR_SLANGC=$build/slangc > /dev/null && cmake --build $build --target NoorRay.Python"
