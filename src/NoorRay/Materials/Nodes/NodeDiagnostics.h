#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace NoorRay::Materials {

//! The warnings of a node compile, and its first failure.
class NodeDiagnostics {
public:
    explicit NodeDiagnostics(bool approximate) : approximate(approximate) {}

    void warn(std::string message);
    //! Fails the compile, or with approximate materials keeps going with a warning.
    void unsupported(const std::string& message);
    //! The first failure sticks.
    void fail(std::string message);
    bool failed() const { return error.has_value(); }
    bool isApproximate() const { return approximate; }
    const std::optional<std::string>& getError() const { return error; }
    std::vector<std::string> takeWarnings() { return std::move(warnings); }

private:
    bool approximate = false;
    std::vector<std::string> warnings;
    std::unordered_set<std::string> seen;
    std::optional<std::string> error;
};

}
