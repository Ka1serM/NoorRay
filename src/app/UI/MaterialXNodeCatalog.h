#pragma once

#include <MaterialXCore/Document.h>

#include <string>
#include <vector>

// One instantiable MaterialX node definition, flattened for the add menu.
// MaterialX ships hundreds of definitions and they differ only in data, so the
// catalog is generated from the standard libraries rather than hand-written.
struct MaterialXNodeType
{
    std::string category;     // MaterialX node category, e.g. "add"
    std::string nodeDefName;  // MaterialX definition name, e.g. "ND_add_color3"
    std::string group;        // Menu heading, from the definition's nodegroup
    std::string outputType;   // Output data type, e.g. "color3"
    std::string label;        // Menu entry text, e.g. "add (color3)"
    std::string documentation;
    std::string searchKey;    // Lowercased category/group/type, for matching
};

// Every node definition in the MaterialX standard libraries, grouped by
// nodegroup and searchable by name. Built once on first use: loading the
// libraries costs a few hundred milliseconds, so it is deferred until the
// editor actually needs the list.
class MaterialXNodeCatalog
{
public:
    static const MaterialXNodeCatalog& instance();

    const std::vector<MaterialXNodeType>& types() const { return types_; }
    // Menu headings, in the order they should be listed.
    const std::vector<std::string>& groups() const { return groups_; }
    // Types matching every whitespace-separated term of the query, in menu
    // order. An empty query matches everything.
    std::vector<const MaterialXNodeType*> search(const std::string& query) const;
    // Types of one group, in menu order.
    std::vector<const MaterialXNodeType*> inGroup(const std::string& group) const;
    // The definition backing a menu entry, or nullptr if it is unavailable.
    MaterialX::ConstNodeDefPtr findNodeDef(const MaterialXNodeType& type) const;
    // Non-empty when the standard libraries could not be loaded, in which case
    // the catalog is empty.
    const std::string& loadError() const { return loadError_; }

private:
    MaterialXNodeCatalog();

    MaterialX::DocumentPtr libraries_;
    std::vector<MaterialXNodeType> types_;
    std::vector<std::string> groups_;
    std::string loadError_;
};
