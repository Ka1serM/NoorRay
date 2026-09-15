#pragma once

// Names a MaterialX document without including MaterialXCore/Document.h and
// with it MaterialX's whole element model. Generated.h only establishes the
// versioned namespace and the MaterialX alias, so headers that merely store or
// pass documents stay cheap; code that inspects one includes Document.h itself.

#include <memory>

#include <MaterialXCore/Generated.h>

MATERIALX_NAMESPACE_BEGIN
class Document;
using DocumentPtr = std::shared_ptr<Document>;
MATERIALX_NAMESPACE_END
