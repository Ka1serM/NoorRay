#pragma once

#include <expected>
#include <string>

#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Tests {

using Result = std::expected<void, std::string>;

Result submit(NoorRHI::Device& device, NoorRHI::CommandList&& list);
Result check(const std::expected<void, NoorRHI::Error>& result);
Result testLoss(NoorRHI::Device& device);
Result testScan(NoorRHI::Device& device);
Result testSort(NoorRHI::Device& device);
Result testAdam(NoorRHI::Device& device);
Result testComparison(NoorRHI::Device& device);
Result testMeshTexture(NoorRHI::Device& device);
Result testSdfMethod(NoorRHI::Device& device);
#ifdef NOORRAY_GAUSSIAN_TRACING
Result testGaussianTracing(NoorRHI::Device& device);
Result testGaussianMethod(NoorRHI::Device& device);
#endif

}
