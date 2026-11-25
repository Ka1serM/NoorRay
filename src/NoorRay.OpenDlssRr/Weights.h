#pragma once

#include <expected>

#include "NoorRay/Core/Error.h"

namespace NoorRay::OpenDlssRr {

//! The weights live in the OpenDLSS-RR folder, where OpenDLSS-RR finds them by size; the first start downloads their range of the NVIDIA library.
class Weights {
public:
    //! Starts the download on its own thread if the folder has no weights.
    static void prefetch();
    [[nodiscard]] static std::expected<void, Core::Error> wait();
};

}
