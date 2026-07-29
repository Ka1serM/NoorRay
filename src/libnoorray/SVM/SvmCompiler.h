#pragma once

// MaterialX document -> SVM bytecode.
//
// Replaces MdlMaterialCompiler (src/libnoorray/MDL, removed) as NoorRay's
// shading-graph compiler. Where MDL asked MaterialX's own MdlShaderGenerator
// to produce MDL source and then had the MDL SDK compile+distill that,
// SvmCompiler walks the MaterialX::Document's node graph directly and emits
// SVM bytecode (SvmTypes.h) -- the same compilation strategy Blender Cycles
// uses for its own ShaderGraph, just with a MaterialX graph as the front end
// instead of Cycles' own ShaderNode object graph (see the port's plan for
// why: materialx_export.py already turns every Blender node into a MaterialX
// subgraph, so a MaterialX-only front end also covers Blender materials).
//
// NoorRay keeps owning BSDF sampling, evaluation, PDFs, spectral upsampling
// and energy conservation (NoorRayCompositeBsdf) -- SvmCompiler only decides
// which lobes to add and with what parameters, exactly as documented for the
// OSL and MDL backends before it.

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// One compiled material's bytecode. Unlike MDL's CompiledMaterialProgram
// (PTX text, one OptiX program group per material topology), this is plain
// data: SvmEval's single fixed interpreter (SvmEval.h) reads it directly, so
// there is no per-material GPU program/module/SBT bookkeeping at all -- see
// Raytracer's simplified material-program table.
struct CompiledSvmProgram
{
    std::vector<std::uint32_t> bytecode;

    // Scene texture registry index per SVM texture slot referenced by
    // NodeImageTexture::textureSlot in `bytecode`, in the order textures
    // were first encountered while compiling. Populated from the caller-
    // supplied resolvedTextures map, same contract as
    // MdlMaterialCompiler::compile() documented for MDL.
    std::vector<std::uint32_t> textureIndices;
};

class SvmCompiler
{
public:
    SvmCompiler() = default;

    SvmCompiler(const SvmCompiler&) = delete;
    SvmCompiler& operator=(const SvmCompiler&) = delete;

    // Compiles one renderable element from a MaterialX document whose
    // standard + noorray definitions are available (via setDataLibrary() or
    // importLibrary()). elementName selects among multiple renderable
    // elements; empty means "the first one found".
    //
    // resolvedTextures: raw MaterialX <image> file path -> Scene texture
    // registry index, already resolved by the caller (hdnoorray/material.cpp
    // via collectImageNodes()), same as MdlMaterialCompiler::compile() took.
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
