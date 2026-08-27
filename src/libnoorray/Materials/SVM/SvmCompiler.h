#pragma once

// MaterialX document -> SVM bytecode.
//
// SvmCompiler walks the MaterialX node graph directly and emits the fixed
// instruction stream consumed by SvmEval.
//
// NoorRay keeps owning BSDF sampling, evaluation, PDFs, spectral upsampling
// and energy conservation; the compiler supplies graph values and closure
// parameters.

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Materials/SVM/SvmTypes.h"

namespace MaterialX_v1_39_4
{
class Document;
using DocumentPtr = std::shared_ptr<Document>;
}
namespace MaterialX = MaterialX_v1_39_4;

namespace nr::svm
{

class SvmCompileError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// One compiled material's bytecode. SvmEval's fixed interpreter reads this
// plain data directly, so materials share one GPU program and program table.
struct CompiledSvmProgram
{
    std::vector<std::uint32_t> bytecode;

    std::uint32_t stackSize{StackSize};

    // Conservative host classification for direct-light candidate discovery.
    // Actual emission remains dynamically evaluated by SVM in the shader.
    bool mayEmit{};

    // Scene texture registry index per SVM texture slot referenced by
    // NodeImageTexture::textureSlot in `bytecode`, in the order textures
    // were first encountered while compiling. Populated from the caller-
    // supplied resolvedTextures map.
    std::vector<std::uint32_t> textureIndices;
};

class SvmCompiler
{
public:
    SvmCompiler() = default;

    SvmCompiler(const SvmCompiler&) = delete;
    SvmCompiler& operator=(const SvmCompiler&) = delete;

    // Compiles one renderable element from a MaterialX document. If the
    // document has no data library attached, the shared NoorRay/MaterialX
    // standard libraries are attached automatically. elementName selects
    // among multiple renderable elements; empty means "the first one found".
    //
    // resolvedTextures: raw MaterialX <image> file path -> Scene texture
    // registry index, already resolved by the caller (hdnoorray/material.cpp
    // via collectImageNodes()).
    //
    // The initial executable slice accepts surfacematerial -> surface,
    // open_pbr_surface/standard_surface, diffuse/dielectric/uniform EDF,
    // and scalar arithmetic, mix, clamp, gamma, blackbody and image value
    // nodes. Unsupported categories are rejected with their MaterialX node
    // name; they are never silently compiled into a different material.
    //
    // Throws SvmCompileError on any failure (unsupported node category,
    // malformed graph, stack exhaustion).
    CompiledSvmProgram compile(
        const MaterialX::DocumentPtr& document, const std::string& elementName = {},
        const std::unordered_map<std::string, std::uint32_t>& resolvedTextures = {});
};

} // namespace nr::svm
