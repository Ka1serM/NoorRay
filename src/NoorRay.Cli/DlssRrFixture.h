#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Render/Upscale/DlssRr.h"

namespace NoorRay::Cli {

//! Writes the DLSS-RR inputs, output and call values of each frame in the layout of OpenDLSS-RR docs/fixtures.md.
class DlssRrFixture {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(const std::filesystem::path& directory);
    //! Call after the frame finished on the GPU.
    [[nodiscard]] std::expected<void, Core::Error> addFrame(const Render::DlssRr& dlssRr);
    [[nodiscard]] std::expected<void, Core::Error> close(const Render::DlssRr& dlssRr);

private:
    std::filesystem::path directory;
    std::vector<std::string> frames;
};

}
