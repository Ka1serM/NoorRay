#pragma once

// MaterialX document utilities shared by the SVM front end.
//
// SVM walks MaterialX graphs directly and does not use a MaterialX shader
// generator. These utilities remain pure document operations.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Materials/Shading/Material.h"

namespace MaterialX_v1_39_4
{
class Document;
using DocumentPtr = std::shared_ptr<Document>;
}
namespace MaterialX = MaterialX_v1_39_4;

namespace nr::materialx
{

// Returns the process-wide MaterialX standard libraries document, loading it
// once on first use. Every ingestion path shares this immutable document so
// they all resolve nodedefs identically without re-parsing the stdlib per
// material (equivalent to the static the standalone paths used to keep).
MaterialX::DocumentPtr getSharedStandardLibraries();

// Returns a canonical default MaterialX surface document: a neutral built-in
// disney_principled node wrapped in a surfacematerial. This is
// what a material is lowered to when it has no authored document at all, and
// what a graph is fallen back to when it no longer compiles.
MaterialX::DocumentPtr defaultMaterial();

// Lowers one authoring struct -- the simple record an importer fills in --
// into a canonical MaterialX surface graph (a built-in disney_principled
// wrapped in a surfacematerial). Importers convert their
// materials through this function
// instead of keeping the authoring struct around: the resulting document is
// what gets stored with the material, so every material in the Scene is a
// real MaterialX document and compiles through the same MaterialX -> SVM
// pipeline as authored .mtlx graphs.
MaterialX::DocumentPtr documentFromAuthoring(const MaterialAuthoring& material);

// Resolves an <image> node's (already filesystem-resolved) file path to
// a slot in the host's texture registry (Scene::getTextures()). The
// caller is expected to have already loaded every image node's file
// (with the color/data encoding appropriate to that node -- something
// only the graph-translation step knows) and captured the resulting
// path -> index mapping in this closure; the resolver only ever
// needs the index. Returns std::nullopt when the texture cannot be
// loaded, in which case the compiled program samples black for it.
using TextureResolver = std::function<std::optional<std::uint32_t>(
    const std::string& resolvedFilePath)>;

// Loads MaterialX's standard library documents (stdlib, pbrlib, bxdf, ...)
// from materialXStdlibDir into a document meant to be attached to user
// documents with Document::setDataLibrary(), or copied with importLibrary(),
// before compile(). Shared by every ingestion path (standalone .mtlx, Hydra)
// so they resolve nodedefs identically.
MaterialX::DocumentPtr loadStandardLibraries(const std::string& materialXStdlibDir);

enum class MaterialXImageColorSpace
{
    Linear,
    Srgb,
};

// One <image> node found by collectImageNodes(), below.
struct MaterialXImageNode
{
    // The node's own "file" input value, exactly as authored -- a path
    // that may be relative to the containing document, or (for a document
    // built in-memory, e.g. by hdNoorRay's Hydra network translator, with
    // already-resolved paths) already absolute. This is also, verbatim,
    // the string a compiled program's TextureResolver will be called with.
    std::string rawFilePath;
    MaterialXImageColorSpace colorSpace{MaterialXImageColorSpace::Linear};
};

// Walks every <image> node in a document, including ones nested inside
// nodegraphs. Callers use this to pre-load every texture a document needs
// (with the color/data encoding this function already knows, since
// TextureResolver only sees a bare filename) before compiling, and to
// build the path -> registry-index table their TextureResolver closes over.
std::vector<MaterialXImageNode> collectImageNodes(const MaterialX::DocumentPtr& document);

} // namespace nr::materialx
