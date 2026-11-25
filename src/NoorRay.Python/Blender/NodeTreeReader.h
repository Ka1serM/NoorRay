#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay/Materials/NodeTree.h"

namespace NoorRay::Python {

//! Reads Blender node trees into NoorRay node trees. A group tree is read once per reader and shared by every tree that uses it.
class NodeTreeReader {
public:
    explicit NodeTreeReader(Reader& reader) : reader(reader) {}

    //! The tree of an owner, empty for none; the caller adds the settings of the owner to its properties.
    std::shared_ptr<Materials::NodeTree> read(const Object& tree);
    //! The Blender node of a node that this reader read; empty for a node that holds no image, IES profile or sky.
    Object find(const Materials::NodeTree& tree, std::uint32_t node) const;
    //! Adds a texture record to a node of a tree that this reader read; a group tree is shared, so its record serves every tree that uses it.
    void addRecord(const Materials::NodeTree& tree, std::uint32_t node, Materials::NodeProperty record);

private:
    struct Setting {
        std::string identifier;
        std::string type;
        std::size_t length = 0;
    };

    void readTree(const Object& source, Materials::NodeTree& tree);
    void readNode(const Object& source, Materials::NodeTree& tree, std::uint32_t index, std::map<std::uintptr_t, std::uint32_t>& groupIndices);
    void readSockets(const Object& collection, std::vector<Materials::NodeSocket>& sockets);
    void readSetting(const Object& source, const Setting& setting, Materials::Node& node);
    void readRamp(const Object& ramp, Materials::Node& node);
    void readCurves(const Object& mapping, Materials::Node& node);
    const std::vector<Setting>& settings(const Object& node, const std::string& idname);
    float worldDistance();

    Reader& reader;
    //! The trees stay for the life of the reader, so that no later tree takes the address of a tree in nodes.
    std::vector<std::shared_ptr<Materials::NodeTree>> roots;
    std::map<std::uintptr_t, std::shared_ptr<Materials::NodeTree>> groups;
    std::map<std::pair<const Materials::NodeTree*, std::uint32_t>, Object> nodes;
    std::map<std::string, std::vector<Setting>> types;
    std::vector<std::string> baseSettings;
    std::optional<float> aoDistance;
};

//! The node tree of an owner that Cycles renders: none where the owner does not use nodes, before Blender 5.
Object nodeTree(Reader& reader, const Object& owner);

Materials::NodeProperty booleanProperty(std::string name, bool value);
Materials::NodeProperty integerProperty(std::string name, std::int64_t value);
Materials::NodeProperty numberProperty(std::string name, float value);
Materials::NodeProperty textProperty(std::string name, std::string value);
Materials::NodeProperty numbersProperty(std::string name, std::span<const float> values);

}
