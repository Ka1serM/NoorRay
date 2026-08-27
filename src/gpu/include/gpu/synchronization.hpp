#pragma once

// The synchronization vocabulary - Stage, GpuToken - lives in types.hpp so
// that every other header can name it without a dependency cycle. This header
// exists so `#include <gpu/synchronization.hpp>` resolves to that vocabulary.
#include "types.hpp"
