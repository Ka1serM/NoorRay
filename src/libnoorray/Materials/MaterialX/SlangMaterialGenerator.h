#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Materials/MaterialX/MaterialXFwd.h"

namespace nr::materialx
{

class NoorRayShaderGenerator;

// A word of a material's parameter block that holds a texture.
struct SlangMaterialTexture
{
    std::uint32_t word{};
    // The image's file path as authored; empty samples white.
    std::string file;
};

// A MaterialX surface generated for the realtime renderer: a Slang module
// whose GeneratedMaterial implements IMaterial
// (Shaders/RealtimeRaytracer/MaterialInterface.slang), and the parameter
// block it reads. Every editable input is a parameter, so documents that
// differ only in their values generate the same source and share a shader.
struct SlangMaterial
{
    // Digest of the generated module. Values and texture bindings are in the
    // parameter block, so they do not create another shader shape.
    std::uint64_t shaderShape{};
    std::string source;
    std::vector<std::uint32_t> parameters;
    // Replaced by the textures' descriptor handles when the block is uploaded.
    std::vector<SlangMaterialTexture> textures;
};

// Generates Slang with MaterialX's Slang generator (MaterialXGenSlang) for a
// NoorRay target that inherits genslang: pattern nodes keep their MaterialX
// implementations, while BSDF, EDF and surface nodes build the closures of
// MaterialInterface.slang instead of lighting the surface. Not thread safe;
// each worker owns one.
class SlangMaterialGenerator
{
public:
    SlangMaterialGenerator();
    ~SlangMaterialGenerator();

    SlangMaterialGenerator(const SlangMaterialGenerator&) = delete;
    SlangMaterialGenerator& operator=(const SlangMaterialGenerator&) = delete;

    // Generates the document's first renderable surface shader. Throws when
    // the document has none or uses a closure the renderer cannot express.
    SlangMaterial generate(const MaterialX::DocumentPtr& document) const;

private:
    MaterialX::DocumentPtr libraries_;
    std::shared_ptr<NoorRayShaderGenerator> generator_;
};

} // namespace nr::materialx
