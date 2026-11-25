#include "DiffRenderer/Methods/Mesh/Remesher.h"

#include <algorithm>
#include <cstdlib>

namespace DiffRenderer::Methods {

namespace {

constexpr int maxRounds = 12;
constexpr int maxFlipRounds = 4;
constexpr int targetValence = 6;
constexpr float minNormalAgreement = 0.2f;
constexpr float relaxation = 0.5f;

std::uint32_t getNextCorner(const std::uint32_t corner) {
    return corner - corner % 3 + (corner + 1) % 3;
}

std::uint32_t getPreviousCorner(const std::uint32_t corner) {
    return corner - corner % 3 + (corner + 2) % 3;
}

Vector3 getNormal(const Vector3 a, const Vector3 b, const Vector3 c) {
    return (b - a).cross(c - a);
}

bool agree(const Vector3 first, const Vector3 second) {
    return first.getNormalized().dot(second.getNormalized()) > minNormalAgreement;
}

int getDeviation(const int valence) {
    return std::abs(valence - targetValence);
}

}

void Remesher::remesh(const float targetLength, const int passes) {
    const float high = targetLength * 4.0f / 3.0f;
    const float low = targetLength * 4.0f / 5.0f;
    link();
    for (int pass = 0; pass < passes; ++pass) {
        for (int round = 0; round < maxRounds; ++round) {
            unlock();
            const int splits = splitLongEdges(high);
            link();
            if (splits == 0)
                break;
        }
        for (int round = 0; round < maxRounds; ++round) {
            unlock();
            const int collapses = collapseShortEdges(low, high);
            removeCollapsed();
            link();
            if (collapses == 0)
                break;
        }
        for (int round = 0; round < maxFlipRounds; ++round) {
            unlock();
            const int flips = flipEdges();
            link();
            if (flips == 0)
                break;
        }
        relax();
    }
}

void Remesher::link() {
    opposites = mesh.findOppositeCorners();
    vertexCorners.assign(mesh.getVertexCount(), 0);
    valences.assign(mesh.getVertexCount(), 0);
    for (std::uint32_t corner = 0; corner < mesh.triangles.size(); ++corner) {
        vertexCorners[mesh.triangles[corner]] = corner;
        ++valences[mesh.triangles[corner]];
    }
    removedTriangles.assign(mesh.getTriangleCount(), 0);
    removedVertices.assign(mesh.getVertexCount(), 0);
}

void Remesher::unlock() {
    lockedVertices.assign(mesh.getVertexCount(), 0);
}

bool Remesher::collectFan(const std::uint32_t vertex, std::vector<std::uint32_t>& fan) const {
    fan.clear();
    if (valences[vertex] == 0)
        return false;
    const std::uint32_t start = vertexCorners[vertex];
    std::uint32_t corner = start;
    do {
        fan.push_back(corner);
        const std::int32_t opposite = opposites[getNextCorner(corner)];
        if (opposite < 0 || fan.size() > valences[vertex])
            return false;
        corner = getNextCorner(static_cast<std::uint32_t>(opposite));
    } while (corner != start);
    return fan.size() == valences[vertex];
}

bool Remesher::isLocked(const std::span<const std::uint32_t> fan) const {
    return std::ranges::any_of(fan, [&](const std::uint32_t corner) {
        return lockedVertices[mesh.triangles[corner]] != 0 || lockedVertices[mesh.triangles[getNextCorner(corner)]] != 0;
    });
}

void Remesher::lock(const std::span<const std::uint32_t> fan) {
    for (const std::uint32_t corner : fan) {
        lockedVertices[mesh.triangles[corner]] = 1;
        lockedVertices[mesh.triangles[getNextCorner(corner)]] = 1;
    }
}

int Remesher::splitLongEdges(const float high) {
    std::vector<std::uint8_t> lockedTriangles(mesh.getTriangleCount(), 0);
    const auto corners = static_cast<std::uint32_t>(mesh.triangles.size());
    int splits = 0;
    for (std::uint32_t corner = 0; corner < corners; ++corner) {
        const std::int32_t opposite = opposites[corner];
        if (opposite < static_cast<std::int32_t>(corner) || lockedTriangles[corner / 3] != 0 || lockedTriangles[static_cast<std::uint32_t>(opposite) / 3] != 0)
            continue;
        const auto other = static_cast<std::uint32_t>(opposite);
        const std::uint32_t a = mesh.triangles[getNextCorner(corner)];
        const std::uint32_t b = mesh.triangles[getPreviousCorner(corner)];
        if ((mesh.positions[a] - mesh.positions[b]).getLength() <= high)
            continue;
        const auto middle = static_cast<std::uint32_t>(mesh.positions.size());
        mesh.positions.push_back((mesh.positions[a] + mesh.positions[b]) * 0.5f);
        mesh.colors.push_back((mesh.colors[a] + mesh.colors[b]) * 0.5f);
        mesh.materials.push_back((mesh.materials[a] + mesh.materials[b]) * 0.5f);
        const std::uint32_t v0 = mesh.triangles[corner];
        const std::uint32_t v1 = mesh.triangles[other];
        mesh.triangles[getPreviousCorner(corner)] = middle;
        mesh.triangles[getPreviousCorner(other)] = middle;
        mesh.triangles.insert(mesh.triangles.end(), {v0, middle, b, v1, middle, a});
        lockedTriangles[corner / 3] = 1;
        lockedTriangles[other / 3] = 1;
        ++splits;
    }
    return splits;
}

bool Remesher::collapse(const std::uint32_t corner, const float high) {
    const std::int32_t opposite = opposites[corner];
    if (opposite < 0)
        return false;
    const auto other = static_cast<std::uint32_t>(opposite);
    const std::uint32_t first = corner / 3;
    const std::uint32_t second = other / 3;
    const std::uint32_t a = mesh.triangles[getNextCorner(corner)];
    const std::uint32_t b = mesh.triangles[getPreviousCorner(corner)];
    if (valences[mesh.triangles[corner]] <= 3 || valences[mesh.triangles[other]] <= 3)
        return false;
    if (!collectFan(a, fanA) || !collectFan(b, fanB) || isLocked(fanA) || isLocked(fanB))
        return false;
    int common = 0;
    for (const std::uint32_t cornerA : fanA)
        for (const std::uint32_t cornerB : fanB)
            common += mesh.triangles[getNextCorner(cornerA)] == mesh.triangles[getNextCorner(cornerB)] ? 1 : 0;
    if (common != 2)
        return false;
    const Vector3 middle = (mesh.positions[a] + mesh.positions[b]) * 0.5f;
    for (const std::vector<std::uint32_t>* fan : {&fanA, &fanB})
        for (const std::uint32_t fanCorner : *fan) {
            const std::uint32_t triangle = fanCorner / 3;
            if (triangle == first || triangle == second)
                continue;
            const Vector3 moved = mesh.positions[mesh.triangles[fanCorner]];
            const Vector3 next = mesh.positions[mesh.triangles[getNextCorner(fanCorner)]];
            const Vector3 previous = mesh.positions[mesh.triangles[getPreviousCorner(fanCorner)]];
            if ((next - middle).getLength() > high || !agree(getNormal(moved, next, previous), getNormal(middle, next, previous)))
                return false;
        }
    lock(fanA);
    lock(fanB);
    for (const std::uint32_t fanCorner : fanB)
        mesh.triangles[fanCorner] = a;
    removedTriangles[first] = 1;
    removedTriangles[second] = 1;
    removedVertices[b] = 1;
    mesh.positions[a] = middle;
    mesh.colors[a] = (mesh.colors[a] + mesh.colors[b]) * 0.5f;
    mesh.materials[a] = (mesh.materials[a] + mesh.materials[b]) * 0.5f;
    return true;
}

int Remesher::collapseShortEdges(const float low, const float high) {
    int collapses = 0;
    for (std::uint32_t corner = 0; corner < mesh.triangles.size(); ++corner) {
        if (opposites[corner] < static_cast<std::int32_t>(corner) || removedTriangles[corner / 3] != 0)
            continue;
        const std::uint32_t a = mesh.triangles[getNextCorner(corner)];
        const std::uint32_t b = mesh.triangles[getPreviousCorner(corner)];
        if (lockedVertices[a] != 0 || lockedVertices[b] != 0 || (mesh.positions[a] - mesh.positions[b]).getLength() >= low)
            continue;
        if (collapse(corner, high))
            ++collapses;
    }
    return collapses;
}

int Remesher::flipEdges() {
    int flips = 0;
    for (std::uint32_t corner = 0; corner < mesh.triangles.size(); ++corner) {
        const std::int32_t opposite = opposites[corner];
        if (opposite < static_cast<std::int32_t>(corner))
            continue;
        const auto other = static_cast<std::uint32_t>(opposite);
        const std::uint32_t a = mesh.triangles[getNextCorner(corner)];
        const std::uint32_t b = mesh.triangles[getPreviousCorner(corner)];
        const std::uint32_t v0 = mesh.triangles[corner];
        const std::uint32_t v1 = mesh.triangles[other];
        if (v0 == v1 || lockedVertices[a] != 0 || lockedVertices[b] != 0 || lockedVertices[v0] != 0 || lockedVertices[v1] != 0)
            continue;
        const auto va = static_cast<int>(valences[a]);
        const auto vb = static_cast<int>(valences[b]);
        const auto v0Valence = static_cast<int>(valences[v0]);
        const auto v1Valence = static_cast<int>(valences[v1]);
        if (va <= 3 || vb <= 3)
            continue;
        const int before = getDeviation(va) + getDeviation(vb) + getDeviation(v0Valence) + getDeviation(v1Valence);
        const int after = getDeviation(va - 1) + getDeviation(vb - 1) + getDeviation(v0Valence + 1) + getDeviation(v1Valence + 1);
        if (after >= before || !collectFan(v0, fanA))
            continue;
        if (std::ranges::any_of(fanA, [&](const std::uint32_t fanCorner) { return mesh.triangles[getNextCorner(fanCorner)] == v1; }))
            continue;
        const Vector3 pa = mesh.positions[a];
        const Vector3 pb = mesh.positions[b];
        const Vector3 p0 = mesh.positions[v0];
        const Vector3 p1 = mesh.positions[v1];
        const Vector3 oldFirst = getNormal(p0, pa, pb);
        const Vector3 oldSecond = getNormal(p1, pb, pa);
        const Vector3 newFirst = getNormal(p0, pa, p1);
        const Vector3 newSecond = getNormal(p1, pb, p0);
        if (!agree(oldFirst, oldSecond) || !agree(newFirst, newSecond) || !agree(newFirst, oldFirst + oldSecond) || !agree(newSecond, oldFirst + oldSecond))
            continue;
        mesh.triangles[getPreviousCorner(corner)] = v1;
        mesh.triangles[getPreviousCorner(other)] = v0;
        --valences[a];
        --valences[b];
        ++valences[v0];
        ++valences[v1];
        for (const std::uint32_t vertex : {a, b, v0, v1})
            lockedVertices[vertex] = 1;
        ++flips;
    }
    return flips;
}

void Remesher::relax() {
    std::vector<Vector3> moved = mesh.positions;
    for (std::uint32_t vertex = 0; vertex < mesh.getVertexCount(); ++vertex) {
        if (!collectFan(vertex, fanA))
            continue;
        const Vector3 position = mesh.positions[vertex];
        Vector3 centroid;
        Vector3 normal;
        for (const std::uint32_t corner : fanA) {
            const Vector3 next = mesh.positions[mesh.triangles[getNextCorner(corner)]];
            centroid += next;
            normal += getNormal(position, next, mesh.positions[mesh.triangles[getPreviousCorner(corner)]]);
        }
        normal = normal.getNormalized();
        const Vector3 offset = centroid * (1.0f / static_cast<float>(fanA.size())) - position;
        moved[vertex] = position + (offset - normal * normal.dot(offset)) * relaxation;
    }
    mesh.positions = std::move(moved);
}

void Remesher::removeCollapsed() {
    std::vector<std::uint32_t> remap(mesh.getVertexCount(), 0);
    std::uint32_t kept = 0;
    for (std::uint32_t vertex = 0; vertex < mesh.getVertexCount(); ++vertex) {
        if (removedVertices[vertex] != 0)
            continue;
        remap[vertex] = kept;
        mesh.positions[kept] = mesh.positions[vertex];
        mesh.colors[kept] = mesh.colors[vertex];
        mesh.materials[kept] = mesh.materials[vertex];
        ++kept;
    }
    mesh.positions.resize(kept);
    mesh.colors.resize(kept);
    mesh.materials.resize(kept);
    std::size_t target = 0;
    for (std::size_t triangle = 0; triangle < removedTriangles.size(); ++triangle) {
        if (removedTriangles[triangle] != 0)
            continue;
        for (std::size_t corner = 0; corner < 3; ++corner)
            mesh.triangles[target * 3 + corner] = remap[mesh.triangles[triangle * 3 + corner]];
        ++target;
    }
    mesh.triangles.resize(target * 3);
}

}
