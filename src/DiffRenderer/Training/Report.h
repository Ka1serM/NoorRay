#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string>

#include "DiffRenderer/Training/Run.h"

namespace DiffRenderer::Training {

//! Writes the runs as JSON: the description and settings of each algorithm, its timings and the history of every step.
[[nodiscard]] std::expected<void, NoorRHI::Error> writeReport(const std::filesystem::path& path, std::span<const Run* const> runs);
std::string getReportJson(std::span<const Run* const> runs);
//! The last sample of each run as one row of a Markdown table.
std::string getReportTable(std::span<const Run* const> runs);

}
