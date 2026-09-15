#pragma once

// Diagnostics for libnoorray. The macros are prefixed because this is a public
// header of an embeddable library: unprefixed LOG_INFO/LOG_ERROR collide with
// almost every host's own logging. Output goes through a single sink so a host
// can capture it instead of having the library write to the process stdout.

#include <sstream>
#include <string_view>

namespace noorray::log {

enum class Level { Debug, Info, Warn, Error, Fatal };

// Receives every formatted message. `message` has no trailing newline and no
// level prefix; the sink decides how to present both.
using Sink = void (*)(Level level, std::string_view message);

// Redirects all subsequent logging. Passing nullptr restores the default sink,
// which writes Debug/Info/Warn to std::cout and Error/Fatal to std::cerr.
void setSink(Sink sink);

void write(Level level, std::string_view message);

} // namespace noorray::log

// The stream is built first and handed over as one string, so a host sink sees
// whole messages rather than interleaved fragments.
#define NR_LOG_AT(level, x)                                                    \
    do {                                                                       \
        std::ostringstream nr_log_stream_;                                     \
        nr_log_stream_ << x;                                                   \
        ::noorray::log::write(level, nr_log_stream_.str());                    \
    } while (false)

#define NR_LOG_INFO(x)  NR_LOG_AT(::noorray::log::Level::Info, x)
#define NR_LOG_WARN(x)  NR_LOG_AT(::noorray::log::Level::Warn, x)
#define NR_LOG_ERROR(x) NR_LOG_AT(::noorray::log::Level::Error, x)
#define NR_LOG_FATAL(x) NR_LOG_AT(::noorray::log::Level::Fatal, x)

#if !defined(NDEBUG)
    #define NR_LOG_DEBUG(x) NR_LOG_AT(::noorray::log::Level::Debug, x)
#else
    #define NR_LOG_DEBUG(x) ((void)0)
#endif
