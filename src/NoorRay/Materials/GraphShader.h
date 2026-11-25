#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "NoorRHI/Pipelines/Pipeline.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Materials/Graph.h"

namespace NoorRay::Materials {

//! The function of Shaders/Materials/Graph.slang whose body the graph of a material kernel replaces.
inline constexpr std::string_view graphFunctionName = "evaluateMaterialGraph";
//! The graph function of the AlphaGraph kernel: it has its own name, because a link joins the donor functions to the host functions of the same name.
inline constexpr std::string_view alphaGraphFunctionName = "evaluateAlphaGraph";
//! The function of the AlphaGraph kernel that a link copies into a kernel, once per alpha graph.
inline constexpr std::string_view alphaGraphDonorName = "alphaGraphCall";
//! The function of Shaders/Materials/Graph.slang whose body a link replaces with a switch over the alpha graphs.
inline constexpr std::string_view alphaGraphSwitchName = "runAlphaGraph";

//! The code of one graph and a key of the code, by which the material kernels are cached.
struct GraphCode {
    std::vector<std::byte> code;
    std::uint64_t key = 0;
};

//! The code of the graph function name that runs graph, as SPIR-V words on Vulkan and as WGSL text on WebGPU.
[[nodiscard]] std::expected<GraphCode, Core::Error> writeGraphCode(const Graph& graph, std::string_view name = graphFunctionName);

//! True when code has the function name, which a kernel keeps only when it calls the function.
[[nodiscard]] std::expected<bool, Core::Error> hasGraphFunction(std::span<const std::byte> code, std::string_view name = graphFunctionName);

//! The shader code with the body of the graph function name from graphCode, which has a function of the same name.
[[nodiscard]] std::expected<std::vector<std::byte>, Core::Error> spliceGraphCode(std::span<const std::byte> code, std::span<const std::byte> graphCode,
                                                                                 std::string_view name = graphFunctionName);

//! base with the graph of code in each of its code variants, kept in storage; variants that share code share the spliced code.
[[nodiscard]] std::expected<void, Core::Error> spliceGraphShader(const NoorRHI::Shader& base, const GraphCode& code,
                                                                 std::array<std::vector<std::byte>, 4>& storage, NoorRHI::Shader& shader,
                                                                 std::string_view name = graphFunctionName);

//! The shader code with a copy of the donor function of each donor, and the alpha graph switch that calls the copy of donor index.
[[nodiscard]] std::expected<std::vector<std::byte>, Core::Error> linkAlphaGraphCode(std::span<const std::byte> code,
                                                                                    std::span<const std::span<const std::byte>> donors);

//! base with the alpha graphs of donors linked into each of its code variants, kept in storage; each donor gives the code of the same variant.
[[nodiscard]] std::expected<void, Core::Error> linkAlphaGraphShader(const NoorRHI::Shader& base, std::span<const NoorRHI::Shader* const> donors,
                                                                    std::array<std::vector<std::byte>, 4>& storage, NoorRHI::Shader& shader);

}
