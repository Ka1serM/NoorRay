#pragma once

#include <string>
#include <vector>

#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay/Materials/Nodes/NodeDiagnostics.h"

namespace NoorRay::Python {

//! The warnings of a sync in one list with those of its node compiles; a failure fails the reader, where the first failure sticks.
class Diagnostics {
public:
    Diagnostics(Reader& reader, bool approximate) : reader(reader), nodes(approximate) {}

    void warn(std::string message) { nodes.warn(std::move(message)); }
    //! Fails the sync, or with approximate materials keeps going with a warning.
    void unsupported(const std::string& message) {
        nodes.unsupported(message);
        settle();
    }
    //! The diagnostics of a node compile, after the reader passed its failure to them, so that the compile stops.
    Materials::NodeDiagnostics& forNodes() {
        if (const std::optional<Core::Error>& error = reader.getError())
            nodes.fail(error->message);
        return nodes;
    }
    //! Passes the failure of a node compile to the reader.
    void settle() {
        if (nodes.failed())
            reader.fail(*nodes.getError());
    }
    std::vector<std::string> takeWarnings() { return nodes.takeWarnings(); }

private:
    Reader& reader;
    Materials::NodeDiagnostics nodes;
};

}
