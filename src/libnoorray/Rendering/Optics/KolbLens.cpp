#include "Rendering/Optics/KolbLens.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nr::optics {
namespace {
struct Glass { SellmeierCoefficients coefficients; };

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    const auto last = value.find_last_not_of(" \t\r\n");
    return first == std::string::npos ? "" : value.substr(first, last - first + 1);
}
std::string upper(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char c) { return std::toupper(c); });
    return value;
}
std::vector<std::string> words(const std::string& line) {
    std::istringstream in(line); std::vector<std::string> result; std::string word;
    while (in >> word) result.push_back(word);
    return result;
}
float number(const std::string& value, const std::string& file, const size_t line) {
    try { return std::stof(value); }
    catch (...) { throw std::runtime_error(file + ":" + std::to_string(line) + ": expected number '" + value + "'"); }
}
std::string utf16leToAscii(const std::string& input) {
    const bool bom = input.size() >= 2 && static_cast<unsigned char>(input[0]) == 0xff
        && static_cast<unsigned char>(input[1]) == 0xfe;
    if (input.size() < 2 || (!bom && input[1] != '\0')) return input;
    std::string output; output.reserve(input.size() / 2);
    const size_t start = bom ? 2 : 0;
    for (size_t i = start; i + 1 < input.size(); i += 2) {
        if (input[i] == '\0' && input[i + 1] == '\0') break;
        if (input[i + 1] != '\0') throw std::runtime_error("Optics catalog contains non-ASCII UTF-16 text");
        output.push_back(input[i]);
    }
    return output;
}
std::unordered_map<std::string, Glass> loadCatalogs(const std::vector<std::string>& paths) {
    std::unordered_map<std::string, Glass> glasses;
    static const std::unordered_set<std::string> catalogMetadata{
        "GC", "ED", "TD", "LD", "OD", "IT", "CC", "CV", "DC", "AZ",
        "BD", "MD", "XR", "FF", "Q", "RA", "RT", "TE", "Y",
        "REPRODUCED"};
    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open AGF catalog '" + path + "'");
        const std::string contents((std::istreambuf_iterator<char>(file)), {});
        std::istringstream input(utf16leToAscii(contents));
        std::string line, name; size_t lineNumber = 0;
        while (std::getline(input, line)) {
            ++lineNumber; const auto fields = words(line); if (fields.empty()) continue;
            const std::string command = upper(fields[0]);
            if (command == "NM" && fields.size() >= 2) name = upper(fields[1]);
            else if (command == "CD" && !name.empty() && fields.size() >= 7) {
                // AGF formula 2 is the usual Schott Sellmeier form: B1 C1 ...
                Glass glass;
                glass.coefficients.b = glm::vec3(number(fields[1], path, lineNumber), number(fields[3], path, lineNumber), number(fields[5], path, lineNumber));
                glass.coefficients.c = glm::vec3(number(fields[2], path, lineNumber), number(fields[4], path, lineNumber), number(fields[6], path, lineNumber));
                glasses[name] = glass;
            }
            else if (catalogMetadata.contains(command)) continue;
            else if (command != "NM" && command != "CD")
                throw std::runtime_error(path + ":" + std::to_string(lineNumber)
                    + ": unsupported AGF command " + command);
        }
    }
    return glasses;
}
}

std::vector<std::string> splitCatalogPaths(const std::string& paths) {
    std::vector<std::string> result; size_t begin = 0;
    while (begin <= paths.size()) { const size_t end = paths.find(';', begin); const auto item = trim(paths.substr(begin, end - begin)); if (!item.empty()) result.push_back(item); if (end == std::string::npos) break; begin = end + 1; }
    return result;
}

ParsedLens loadZmx(const std::string& lensPath, const std::vector<std::string>& catalogPaths) {
    const auto glasses = loadCatalogs(catalogPaths);
    std::ifstream file(lensPath, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open ZMX lens '" + lensPath + "'");
    const std::string input((std::istreambuf_iterator<char>(file)), {});
    std::istringstream in(utf16leToAscii(input));
    struct Raw { float curvature{}, thickness{}, semiDiameter{}; float conic{}; std::array<float, MaxAsphereTerms> asphere{}; std::string glass; bool stop{}; };
    std::vector<Raw> raw; Raw* current = nullptr; std::string line; size_t lineNumber = 0;
    while (std::getline(in, line)) {
        ++lineNumber; const auto f = words(line); if (f.empty()) continue;
        const std::string command = upper(f[0]);
        if (command == "SURF") { raw.emplace_back(); current = &raw.back(); continue; }
        // ZMX contains a substantial amount of editor metadata. These records
        // are safe to ignore because they do not alter sequential geometry;
        // every other command is rejected instead of silently changing the
        // optical model.
        static const std::unordered_set<std::string> metadata{
            "VERS", "MODE", "NAME", "NOTE", "PFIL", "LANG", "UNIT", "ENPD",
            "ENVD", "GFAC", "GCAT", "RAIM", "PUSH", "SDMA", "OMMA", "FTYP",
            "ROPD", "PICB", "XFLN", "YFLN", "FWGN", "VDXN", "VDYN", "VCXN",
            "VCYN", "VANN", "WAVM", "PWAV", "POLS", "GLRS", "GSTD", "NSCD",
            "COFN", "COMM", "FIMP", "HIDE", "VDSZ", "POPS", "FLAP", "COAT",
            "DUMMY", "IGNR", "SSID", "MIRR", "SLAB", "MEMA", "BLNK",
            "OPDX", "LUID", "EERA", "DMFS", "TOL", "MNUM", "MOFF", "AUTH",
            "IWDP", "HYPR"};
        if (!current) {
            if (metadata.contains(command)) continue;
            throw std::runtime_error(lensPath + ":" + std::to_string(lineNumber)
                + ": unsupported global ZMX command " + command);
        }
        if (command == "CURV" && f.size() >= 2) current->curvature = number(f[1], lensPath, lineNumber);
        else if (command == "DISZ" && f.size() >= 2) current->thickness = number(f[1], lensPath, lineNumber);
        else if ((command == "DIAM" || command == "MEMA") && f.size() >= 2) current->semiDiameter = number(f[1], lensPath, lineNumber);
        else if (command == "GLAS" && f.size() >= 2) current->glass = upper(f[1]);
        else if (command == "TYPE" && f.size() >= 2) { const auto type = upper(f[1]); if (type != "STANDARD" && type != "EVENASPH" && type != "EVENASPHERE" && type != "XASPHERE") throw std::runtime_error(lensPath + ":" + std::to_string(lineNumber) + ": unsupported surface type " + type); }
        else if (command == "CONI" && f.size() >= 2) current->conic = number(f[1], lensPath, lineNumber);
        else if (command == "XDAT" && f.size() >= 3) {
            const int term = std::stoi(f[1]);
            // XASPHERE stores radius/conic in the first records and the
            // even-power polynomial coefficients from record four onward.
            if (term >= 4 && term < 4 + static_cast<int>(MaxAsphereTerms))
                current->asphere[term - 4] = number(f[2], lensPath, lineNumber);
            else if (term > 12)
                throw std::runtime_error(lensPath + ":" + std::to_string(lineNumber) + ": unsupported XASPHERE term");
        }
        else if (command == "PARM" && f.size() >= 3) { const int i = std::stoi(f[1]); if (i == 1) current->conic = number(f[2], lensPath, lineNumber); else if (i >= 2 && i < 2 + static_cast<int>(MaxAsphereTerms)) current->asphere[i - 2] = number(f[2], lensPath, lineNumber); else throw std::runtime_error(lensPath + ":" + std::to_string(lineNumber) + ": unsupported asphere term"); }
        else if (command == "STOP") current->stop = true;
        else if (metadata.contains(command)) continue;
        else throw std::runtime_error(lensPath + ":" + std::to_string(lineNumber)
            + ": unsupported ZMX command " + command);
    }
    if (raw.size() < 3) throw std::runtime_error(lensPath + ": lens has fewer than two optical surfaces");
    ParsedLens result; LensSnapshot& lens = result.snapshot; lens.media[0].sellmeier = constantIorSellmeier(1.f);
    // ZMX is object-to-image.  Discard its object plane, reverse all remaining
    // interfaces, and place the image-side vertex at positive z from the film.
    std::vector<float> z(raw.size()); for (size_t i = 1; i < raw.size(); ++i) z[i] = z[i - 1] + raw[i - 1].thickness;
    const float imageZ = z.back(); uint32_t medium = 0;
    for (size_t backwards = raw.size() - 1; backwards > 0; --backwards) {
        if (lens.surfaceCount == MaxLensSurfaces) throw std::runtime_error(lensPath + ": too many optical surfaces");
        const Raw& r = raw[backwards]; Surface& s = lens.surfaces[lens.surfaceCount++];
        s.z = imageZ - z[backwards]; s.radius = fabsf(r.curvature) > 1e-8f ? -1.f / r.curvature : 0.f;
        s.apertureRadius = std::max(0.001f, r.semiDiameter); s.conic = r.conic; s.asphere = r.asphere; s.isStop = r.stop;
        if (!r.glass.empty() && r.glass != "AIR") { const auto it = glasses.find(r.glass); if (it == glasses.end()) throw std::runtime_error(lensPath + ": glass '" + r.glass + "' is absent from the supplied AGF catalogs"); lens.media[++medium].sellmeier = it->second.coefficients; }
        else { lens.media[++medium].sellmeier = constantIorSellmeier(1.f); }
        s.mediumAfter = medium;
    }
    lens.mediumCount = medium + 1; lens.rearPupilZ = lens.surfaces[0].z; lens.rearPupilRadius = lens.surfaces[0].apertureRadius;
    // A compact paraxial estimate is stable metadata even when the ZMX file
    // does not carry an explicit EFL. Summing surface powers is preferable to
    // using the final (often planar) vertex radius and remains deterministic
    // for the compact GPU snapshot.
    float opticalPower = 0.0f;
    uint32_t incomingMedium = 0;
    for (uint32_t i = 0; i < lens.surfaceCount; ++i) {
        const Surface& surface = lens.surfaces[i];
        const uint32_t outgoingMedium = std::min(surface.mediumAfter, lens.mediumCount - 1);
        const float nIncoming = sellmeierIor(lens.media[incomingMedium].sellmeier, FraunhoferGreenNm);
        const float nOutgoing = sellmeierIor(lens.media[outgoingMedium].sellmeier, FraunhoferGreenNm);
        if (std::fabs(surface.radius) > 1e-6f)
            opticalPower += (nOutgoing - nIncoming) / surface.radius;
        incomingMedium = outgoingMedium;
    }
    if (std::fabs(opticalPower) > 1e-6f)
        lens.focalLengthMm = std::fabs(1.0f / opticalPower);
    else {
        lens.focalLengthMm = 0.1f;
        for (uint32_t i = 0; i < lens.surfaceCount; ++i)
            if (std::fabs(lens.surfaces[i].radius) > 1e-6f)
                lens.focalLengthMm = std::max(0.1f, std::fabs(lens.surfaces[i].radius));
    }
    result.message = std::to_string(lens.surfaceCount) + " native surfaces";
    return result;
}
} // namespace nr::optics
