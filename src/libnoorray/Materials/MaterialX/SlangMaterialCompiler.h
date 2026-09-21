#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nr::materialx
{

// SPIR-V of one generated material's callable shader, entry point "main".
struct MaterialShader
{
    std::string source;
    std::vector<std::uint32_t> spirv;
};

// Compiles generated material modules (SlangMaterialGenerator) against
// MaterialInterface.slang with the Slang compiler library. Cross-worker
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
