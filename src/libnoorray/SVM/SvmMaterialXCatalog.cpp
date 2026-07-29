#include "SVM/SvmMaterialXCatalog.h"

#include <algorithm>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Interface.h>
#include <MaterialXCore/Node.h>

#include "MaterialX/MaterialXCompiler.h"

namespace nr::svm
{
namespace
{
MaterialXPortSignature portSignature(const MaterialX::ValueElementPtr& port)
{
    return {port->getName(), port->getType(), port->getValueString()};
}
} // namespace

SvmMaterialXCatalog SvmMaterialXCatalog::load(const std::string& materialXLibraryDir)
{
    SvmMaterialXCatalog catalog;
    const MaterialX::DocumentPtr library = nr::materialx::loadStandardLibraries(materialXLibraryDir);
    for (const MaterialX::NodeDefPtr& definition : library->getNodeDefs()) {
        if (!definition->hasNodeString())
            continue;
        MaterialXNodeSignature signature;
        signature.nodeDefName = definition->getName();
        signature.category = definition->getNodeString();
        signature.outputType = definition->getType();
        signature.nodeGroup = definition->getNodeGroup();
        signature.sourceUri = definition->getActiveSourceUri();
        for (const MaterialX::InterfaceElementPtr& implementation :
             library->getMatchingImplementations(definition->getName())) {
            MaterialXImplementationSignature implementationSignature;
            implementationSignature.target = implementation->getTarget();
            implementationSignature.category = implementation->getCategory();
            implementationSignature.name = implementation->getName();
            implementationSignature.sourceUri = implementation->getActiveSourceUri();
            if (const MaterialX::ImplementationPtr code = implementation->asA<MaterialX::Implementation>()) {
                implementationSignature.file = code->getFile();
                implementationSignature.function = code->getFunction();
                // sourcecode is a standard MaterialX implementation
                // attribute; unlike file/function it has no typed accessor.
                implementationSignature.sourceCode = code->getAttribute("sourcecode");
            }
            signature.implementations.push_back(std::move(implementationSignature));
        }
        for (const MaterialX::InputPtr& input : definition->getActiveInputs())
            signature.inputs.push_back(portSignature(input));
        for (const MaterialX::OutputPtr& output : definition->getActiveOutputs())
            signature.outputs.push_back(portSignature(output));
        catalog.nodeDefs_.push_back(std::move(signature));
    }
    std::ranges::sort(catalog.nodeDefs_, {}, &MaterialXNodeSignature::nodeDefName);
    return catalog;
}

std::vector<const MaterialXNodeSignature*> SvmMaterialXCatalog::find(
    const std::string& category) const
{
    std::vector<const MaterialXNodeSignature*> result;
    for (const MaterialXNodeSignature& signature : nodeDefs_) {
        if (signature.category == category)
            result.push_back(&signature);
    }
    return result;
}

} // namespace nr::svm
