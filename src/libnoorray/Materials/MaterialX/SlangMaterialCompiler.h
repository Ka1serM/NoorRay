#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nr::materialx
{

// SPIR-V of one generated material's hit stages (MaterialHit.slang), each
// with the entry point named after its member.
struct MaterialShader
{
    std::string source;
    // Evaluates the material into the realtime hit payload.
    std::vector<std::uint32_t> closestHit;
    // Apply the material's opacity to surface and shadow rays.
    std::vector<std::uint32_t> anyHit;
    std::vector<std::uint32_t> shadowAnyHit;
};

// Compiles generated material modules (SlangMaterialGenerator) against
// MaterialInterface.slang and MaterialHit.slang with the Slang compiler
// library. Cross-worker
// shader-shape sharing is owned by MaterialXSceneRuntime. Not thread safe;
// each worker owns one.
class SlangMaterialCompiler
{
public:
    SlangMaterialCompiler();
    ~SlangMaterialCompiler();

    SlangMaterialCompiler(const SlangMaterialCompiler&) = delete;
    SlangMaterialCompiler& operator=(const SlangMaterialCompiler&) = delete;

    // Throws with Slang's diagnostics when the source does not compile.
    std::shared_ptr<const MaterialShader> compile(const std::string& source);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nr::materialx
