#include "NoorRay/Materials/Nodes/NodeDiagnostics.h"

namespace NoorRay::Materials {

void NodeDiagnostics::warn(std::string message) {
    if (seen.insert(message).second)
        warnings.push_back(std::move(message));
}

void NodeDiagnostics::unsupported(const std::string& message) {
    if (!approximate) {
        fail(message + "; turn off Strict Materials to ignore it");
        return;
    }
    warn(message + "; it is ignored");
}

void NodeDiagnostics::fail(std::string message) {
    if (!error)
        error = std::move(message);
}

}
