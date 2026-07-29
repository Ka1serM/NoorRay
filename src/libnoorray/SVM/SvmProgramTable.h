#pragma once

#include <cstdint>
#include <vector>

#include "SVM/SvmCompiler.h"
#include "SVM/SvmTypes.h"

namespace nr::svm
{
// Mirrors Cycles' ShaderManager::svm_nodes storage model: all shaders share
// one word array and each material stores the start of its instruction range.
// Replacing a material appends a new immutable range; old ranges remain valid
// until the next scene-wide rebuild, so background compilation never races a
// render using the previous program.
class SvmProgramTable
{
public:
    std::uint32_t append(const CompiledSvmProgram& program);
    SvmProgramRecord replace(std::uint32_t index, const CompiledSvmProgram& program);
    void clear();

    const std::vector<std::uint32_t>& words() const { return words_; }
    const std::vector<std::uint32_t>& textureIndices() const { return textureIndices_; }
    const std::vector<SvmProgramRecord>& records() const { return records_; }

private:
    std::vector<std::uint32_t> words_;
    std::vector<std::uint32_t> textureIndices_;
    std::vector<SvmProgramRecord> records_;
};
} // namespace nr::svm
