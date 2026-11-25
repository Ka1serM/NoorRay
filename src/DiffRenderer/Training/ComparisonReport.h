#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "DiffRenderer/Training/Comparison.h"

namespace DiffRenderer::Training {

//! The dataset, the budget and, for each method, its final metrics, statistics, evaluations, test views and the loss and time of every iteration.
std::string getComparisonJson(const Comparison& comparison);
//! One row for each method, with its final metrics.
std::string getComparisonTable(const Comparison& comparison);
//! Writes report.json and report.md into the output folder of the comparison.
[[nodiscard]] std::expected<void, NoorRHI::Error> writeComparisonReport(const Comparison& comparison);

}
