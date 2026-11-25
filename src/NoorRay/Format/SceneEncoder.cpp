#include "NoorRay/Format/SceneEncoder.h"

#include <sstream>
#include <utility>

namespace NoorRay::Format {

std::uint32_t SharedObjects::Table::add(const void* object, std::string bytes) {
    const auto index = static_cast<std::uint32_t>(indicesByBytes.size());
    return indices[object] = indicesByBytes.try_emplace(std::move(bytes), index).first->second;
}

void SharedObjects::write(BinaryWriter& writer) const {
    for (const Table* table : {&graphs, &programs, &nodeTrees}) {
        std::vector<const std::string*> ordered(table->indicesByBytes.size());
        for (const auto& [bytes, index] : table->indicesByBytes)
            ordered[index] = &bytes;
        writer.write(std::uint64_t{ordered.size()});
        for (const std::string* bytes : ordered)
            writer.writeArray(std::span<const char>(*bytes));
    }
}

void SceneEncoder::transferTopology(const Scene::Mesh& mesh) {
    writer.write(mesh.topology != nullptr);
    if (!mesh.topology)
        return;
    encode(mesh.topology->getNormalSource());
    encode(mesh.topology->getCornerVertices());
    encode(mesh.topology->getFaceOffsets());
}

//! The bytes of the groups of a node tree are in the table before its own bytes, so the reader finds them.
template <class Object, class Source>
void SceneEncoder::encodeShared(SharedObjects::Table& table, const Object* object, const Source& getSource) {
    if (!object) {
        writer.write(UINT32_MAX);
        return;
    }
    if (const auto known = table.indices.find(object); known != table.indices.end()) {
        writer.write(known->second);
        return;
    }
    std::ostringstream stream(std::ios::binary);
    BinaryWriter objectWriter(stream);
    SceneEncoder(objectWriter, scene, shared)(getSource(*object));
    writer.write(table.add(object, std::move(stream).str()));
}

void SceneEncoder::encode(const std::shared_ptr<const Materials::Graph>& graph) {
    encodeShared(shared.graphs, graph.get(), [](const Materials::Graph& value) -> const Materials::Graph::Source& { return value.getSource(); });
}

void SceneEncoder::encode(const std::shared_ptr<const Materials::Program>& program) {
    encodeShared(shared.programs, program.get(), [](const Materials::Program& value) {
        const auto copy = [](const auto values) { return std::vector(values.begin(), values.end()); };
        return Materials::Program::Source{copy(value.getRegisters()), copy(value.getInstructions()), copy(value.getParameters()), copy(value.getAttributes()),
                                          value.getDisplacement()};
    });
}

void SceneEncoder::encode(const std::shared_ptr<const Materials::NodeTree>& tree) {
    encodeShared(shared.nodeTrees, tree.get(), [](const Materials::NodeTree& value) -> const Materials::NodeTree& { return value; });
}

}
