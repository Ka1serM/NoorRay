#include "Logging/Log.h"

#include <iostream>

namespace noorray::log {
namespace {

const char* prefix(const Level level) {
    switch (level) {
        case Level::Debug: return "[DEBUG] ";
        case Level::Info:  return "[INFO] ";
        case Level::Warn:  return "[WARN] ";
        case Level::Error: return "[ERROR] ";
        case Level::Fatal: return "[FATAL] ";
    }
    return "";
}

void defaultSink(const Level level, const std::string_view message) {
    std::ostream& out = level == Level::Error || level == Level::Fatal
        ? std::cerr : std::cout;
    out << prefix(level) << message << std::endl;
}

Sink g_sink = &defaultSink;

} // namespace

void setSink(const Sink sink) { g_sink = sink ? sink : &defaultSink; }

void write(const Level level, const std::string_view message) { g_sink(level, message); }

} // namespace noorray::log
