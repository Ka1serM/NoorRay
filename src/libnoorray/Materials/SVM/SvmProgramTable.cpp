#include "Materials/SVM/SvmProgramTable.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace nr::svm
{
std::uint32_t SvmProgramTable::append(const CompiledSvmProgram& program)
{
    const auto checkedSize = [](const std::size_t size, const char* what) {
        if (size > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error(std::string("SVM ") + what + " exceeds 32-bit GPU addressing");
        return static_cast<std::uint32_t>(size);
    };
    SvmProgramRecord record{
        checkedSize(words_.size(), "word buffer"),
        checkedSize(program.bytecode.size(), "program"),
        checkedSize(textureIndices_.size(), "texture buffer"),
        checkedSize(program.textureIndices.size(), "texture list")};
    words_.insert(words_.end(), program.bytecode.begin(), program.bytecode.end());
    textureIndices_.insert(textureIndices_.end(), program.textureIndices.begin(), program.textureIndices.end());
    records_.push_back(record);
    return static_cast<std::uint32_t>(records_.size() - 1);
}

SvmProgramRecord SvmProgramTable::replace(const std::uint32_t index,
    const CompiledSvmProgram& program)
{
    if (index >= records_.size())
        throw std::out_of_range("SVM program record index out of range");

    // Keep previously uploaded ranges alive until the next scene-wide clear,
    // then atomically repoint this material's record to the appended code.
    // Use an append-and-repoint lifetime model for
    // shader SVM offsets while a scene update is being prepared.
    const std::uint32_t replacement = append(program);
    records_[index] = records_[replacement];
    const SvmProgramRecord result = records_[index];
    records_.pop_back();
    return result;
}

void SvmProgramTable::clear()
{
    words_.clear();
    textureIndices_.clear();
    records_.clear();
}
} // namespace nr::svm
