#include "NoorRay/Materials/GraphShader.h"

#include <algorithm>
#include <format>
#include <utility>

#include "NoorRay/Core/Log.h"

namespace NoorRay::Materials {

std::expected<void, Core::Error> spliceGraphShader(const NoorRHI::Shader& base, const GraphCode& code, std::array<std::vector<std::byte>, 4>& storage,
                                                   NoorRHI::Shader& shader, const std::string_view name) {
    shader = base;
    for (std::size_t variant = 0; variant < base.code.size(); ++variant) {
        if (base.code[variant].empty())
            continue;
        const auto same = std::ranges::find_if(base.code.begin(), base.code.begin() + static_cast<std::ptrdiff_t>(variant),
                                               [&](const std::span<const std::byte> other) { return other.data() == base.code[variant].data(); });
        if (same != base.code.begin() + static_cast<std::ptrdiff_t>(variant)) {
            shader.code[variant] = shader.code[static_cast<std::size_t>(same - base.code.begin())];
            continue;
        }
        std::expected<std::vector<std::byte>, Core::Error> spliced = spliceGraphCode(base.code[variant], code.code, name);
        if (!spliced) {
            Core::Log::error("adding the material graph {:016x} to {} failed: {}", code.key, base.entryPoint, spliced.error().message);
            return std::unexpected(std::move(spliced.error()));
        }
        storage[variant] = std::move(*spliced);
        shader.code[variant] = storage[variant];
    }
    return {};
}

std::expected<void, Core::Error> linkAlphaGraphShader(const NoorRHI::Shader& base, const std::span<const NoorRHI::Shader* const> donors,
                                                      std::array<std::vector<std::byte>, 4>& storage, NoorRHI::Shader& shader) {
    shader = base;
    for (std::size_t variant = 0; variant < base.code.size(); ++variant) {
        if (base.code[variant].empty())
            continue;
        const auto same = std::ranges::find_if(base.code.begin(), base.code.begin() + static_cast<std::ptrdiff_t>(variant),
                                               [&](const std::span<const std::byte> other) { return other.data() == base.code[variant].data(); });
        if (same != base.code.begin() + static_cast<std::ptrdiff_t>(variant)) {
            shader.code[variant] = shader.code[static_cast<std::size_t>(same - base.code.begin())];
            continue;
        }
        std::vector<std::span<const std::byte>> codes;
        for (const NoorRHI::Shader* donor : donors) {
            if (donor->code[variant].empty())
                return Core::fail(Core::ErrorCode::InvalidState, std::format("the alpha graph kernel has no code variant {} for {}", variant, base.entryPoint));
            codes.push_back(donor->code[variant]);
        }
        std::expected<std::vector<std::byte>, Core::Error> linked = linkAlphaGraphCode(base.code[variant], codes);
        if (!linked) {
            Core::Log::error("linking {} alpha graphs into {} failed: {}", donors.size(), base.entryPoint, linked.error().message);
            return std::unexpected(std::move(linked.error()));
        }
        storage[variant] = std::move(*linked);
        shader.code[variant] = storage[variant];
    }
    return {};
}

}
