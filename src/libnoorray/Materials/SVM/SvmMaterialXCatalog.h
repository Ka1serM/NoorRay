#pragma once

#include <memory>
#include <string>
#include <vector>

namespace MaterialX_v1_39_4
{
class Document;
using DocumentPtr = std::shared_ptr<Document>;
}
namespace MaterialX = MaterialX_v1_39_4;

namespace nr::svm
{

// A lossless view of a MaterialX nodedef interface. The SVM compiler uses
// this as its conformance source: MaterialX owns node names, port types,
// defaults and implementation associations; SVM owns fixed bytecode lowering.
struct MaterialXPortSignature
{
    std::string name;
    std::string type;
    std::string defaultValue;
};

struct MaterialXImplementationSignature
{
    std::string target;
    std::string category;
    std::string name;
    std::string file;
    std::string function;
    std::string sourceCode;
    std::string sourceUri;
};

struct MaterialXNodeSignature
{
    std::string nodeDefName;
    std::string category;
    std::string outputType;
    std::string nodeGroup;
    std::string sourceUri;
    std::vector<MaterialXPortSignature> inputs;
    std::vector<MaterialXPortSignature> outputs;
    std::vector<MaterialXImplementationSignature> implementations;
};

class SvmMaterialXCatalog
{
public:
    // Introspect the exact MaterialX libraries vendored with this checkout.
    // This is intentionally library driven rather than a handwritten list,
    // so MaterialX upgrades surface as coverage changes.
    static SvmMaterialXCatalog load(const std::string& materialXLibraryDir);

    const std::vector<MaterialXNodeSignature>& nodeDefs() const { return nodeDefs_; }
    std::vector<const MaterialXNodeSignature*> find(const std::string& category) const;

private:
    std::vector<MaterialXNodeSignature> nodeDefs_;
};

} // namespace nr::svm
