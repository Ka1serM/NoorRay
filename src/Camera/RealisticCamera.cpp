#include "RealisticCamera.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <imgui.h>
#include <limits>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include "glm/gtx/norm.hpp"
#include "Scene/Scene.h"
#include "UI/ImGuiManager.h"

namespace {
std::string readTextFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return {};
    std::ostringstream stream;
    stream << file.rdbuf();
    std::string text = stream.str();
    text.erase(std::remove(text.begin(), text.end(), '\0'), text.end());
    return text;
}

std::string lowerCopy(std::string value)
{
    std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string trimCopy(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> splitTokens(const std::string& line)
{
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token)
        tokens.push_back(token);
    return tokens;
}

bool parseFloat(const std::string& token, float& value)
{
    const char* begin = token.data();
    const char* end = token.data() + token.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr != begin;
}

bool parseJsonNumber(const std::string& text, const char* key, float& value)
{
    const std::regex pattern(std::string("\"") + key + R"("\s*:\s*([-+0-9.eE]+))");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
        return false;
    return parseFloat(match[1].str(), value);
}

bool parseJsonResolution(const std::string& text, uint32_t& width, uint32_t& height)
{
    const std::regex pattern(R"("resolution"\s*:\s*\{[^}]*"width"\s*:\s*([0-9]+)[^}]*"height"\s*:\s*([0-9]+)[^}]*\})");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
        return false;

    float parsedWidth = 0.0f;
    float parsedHeight = 0.0f;
    if (!parseFloat(match[1].str(), parsedWidth) || !parseFloat(match[2].str(), parsedHeight))
        return false;
    if (parsedWidth < 1.0f || parsedHeight < 1.0f)
        return false;

    width = static_cast<uint32_t>(std::round(parsedWidth));
    height = static_cast<uint32_t>(std::round(parsedHeight));
    return true;
}

std::string normalizeGlassName(std::string value)
{
    value = trimCopy(value);
    std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

std::vector<std::string> splitPathList(const std::string& paths)
{
    std::vector<std::string> result;
    std::string current;
    std::istringstream stream(paths);
    while (std::getline(stream, current, ';')) {
        size_t start = 0;
        while (start < current.size()) {
            const size_t comma = current.find(',', start);
            std::string path = trimCopy(current.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
            if (!path.empty())
                result.push_back(path);
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
    }
    return result;
}

std::string joinPaths(const std::vector<std::string>& paths)
{
    std::string result;
    for (const std::string& path : paths) {
        if (!result.empty())
            result += ';';
        result += path;
    }
    return result;
}

std::unordered_map<std::string, float> loadAgfCatalogs(const std::string& paths)
{
    std::unordered_map<std::string, float> catalog;
    for (const std::string& path : splitPathList(paths)) {
        std::istringstream stream(readTextFile(path));
        std::string line;
        while (std::getline(stream, line)) {
            const auto tokens = splitTokens(trimCopy(line));
            if (tokens.size() < 5 || tokens[0] != "NM")
                continue;
            float ior = 0.0f;
            if (parseFloat(tokens[4], ior) && ior > 1.0f)
                catalog[normalizeGlassName(tokens[1])] = ior;
        }
    }
    return catalog;
}

struct CpuRay {
    vec3 origin{};
    vec3 direction{};
};

float selectSurfaceT(float radius, const CpuRay& ray, float t0, float t1)
{
    const bool useCloserT = (ray.direction.z > 0.0f) != (radius < 0.0f);
    return useCloserT ? std::min(t0, t1) : std::max(t0, t1);
}

bool intersectRossSphere(const RealisticLensElement& element, float surfaceOffset, const CpuRay& ray, float& t, vec3& normal)
{
    t = 0.0f;
    normal = vec3(0.0f, 0.0f, 1.0f);
    const float surfaceCenter = element.vertexZ + surfaceOffset;
    const float zCenter = -surfaceCenter + element.radius;
    if (std::abs(element.radius) <= 1e-7f) {
        if (std::abs(ray.direction.z) <= 1e-7f)
            return false;
        t = (-surfaceCenter - ray.origin.z) / ray.direction.z;
        normal = ray.direction.z > 0.0f ? vec3(0.0f, 0.0f, -1.0f) : vec3(0.0f, 0.0f, 1.0f);
        return t >= 0.0f;
    }

    const vec3 o = ray.origin - vec3(0.0f, 0.0f, zCenter);
    const float a = glm::dot(ray.direction, ray.direction);
    const float b = 2.0f * glm::dot(ray.direction, o);
    const float c = glm::dot(o, o) - element.radius * element.radius;
    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return false;

    const float root = std::sqrt(discriminant);
    const float invDenom = 1.0f / (2.0f * a);
    const float t0 = (-b - root) * invDenom;
    const float t1 = (-b + root) * invDenom;
    t = selectSurfaceT(element.radius, ray, t0, t1);
    if (t < 0.0f)
        return false;

    normal = glm::normalize(o + ray.direction * t);
    if (glm::dot(normal, -ray.direction) < 0.0f)
        normal = -normal;
    return true;
}

bool refractRoss(const vec3& point, vec3 normal, const CpuRay& ray, float iorIn, float iorOut, CpuRay& result)
{
    normal = -normal;
    float cosThetaI = glm::dot(normal, ray.direction);
    float eta = iorIn / iorOut;
    if (cosThetaI < 0.0f) {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
        normal = -normal;
    }

    const float sin2ThetaI = std::max(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f)
        return false;

    const float cosThetaT = std::sqrt(1.0f - sin2ThetaT);
    const vec3 f = -ray.direction / eta;
    const vec3 f1 = normal * (cosThetaI / eta - cosThetaT);
    result.origin = point;
    result.direction = glm::normalize(f + f1);
    return true;
}

bool traceFromFilmRoss(const std::vector<RealisticLensElement>& elements, float surfaceOffset, const CpuRay& inputRay, CpuRay& outputRay)
{
    if (elements.empty())
        return false;

    // Public realistic-camera local space matches ROSS: film at z=0 and +Z
    // travels from film toward the world. ROSS's sequential tracer flips Z
    // internally and stores surface centers at negative Z during intersection.
    CpuRay tracedRay{{inputRay.origin.x, inputRay.origin.y, -inputRay.origin.z},
                     {inputRay.direction.x, inputRay.direction.y, -inputRay.direction.z}};

    for (int i = static_cast<int>(elements.size()) - 1; i >= 0; --i) {
        const RealisticLensElement& element = elements[static_cast<size_t>(i)];
        const float surfaceCenter = element.vertexZ + surfaceOffset;
        if (element.isAperture != 0) {
            if (std::abs(tracedRay.direction.z) <= 1e-7f)
                return false;
            const float t = (-surfaceCenter - tracedRay.origin.z) / tracedRay.direction.z;
            if (t < 0.0f)
                return false;
            const vec3 p = tracedRay.origin + tracedRay.direction * t;
            if (glm::length2(vec2(p.x, p.y)) > element.apertureRadius * element.apertureRadius)
                return false;
            continue;
        }

        float t = 0.0f;
        vec3 normal{};
        if (!intersectRossSphere(element, surfaceOffset, tracedRay, t, normal))
            return false;

        const vec3 point = tracedRay.origin + tracedRay.direction * t;
        if (glm::length2(vec2(point.x, point.y)) > element.apertureRadius * element.apertureRadius)
            return false;

        const float etaI = element.ior;
        const float etaT = (i > 0 && elements[static_cast<size_t>(i - 1)].ior != 0.0f)
            ? elements[static_cast<size_t>(i - 1)].ior
            : 1.0f;

        CpuRay refracted{};
        if (!refractRoss(point, normal, tracedRay, etaT, etaI, refracted))
            return false;
        tracedRay = {refracted.origin, -refracted.direction};
    }

    // Convert the internally flipped ROSS ray back to the public +Z-forward
    // lens space before mapping it onto Noor's camera basis.
    outputRay = {{tracedRay.origin.x, tracedRay.origin.y, -tracedRay.origin.z},
                 {tracedRay.direction.x, tracedRay.direction.y, -tracedRay.direction.z}};
    return true;
}

bool traceFromSceneRoss(const std::vector<RealisticLensElement>& elements, float surfaceOffset, const CpuRay& inputRay, CpuRay& outputRay)
{
    if (elements.empty())
        return false;

    CpuRay tracedRay{{inputRay.origin.x, inputRay.origin.y, -inputRay.origin.z},
                     {inputRay.direction.x, inputRay.direction.y, -inputRay.direction.z}};

    for (size_t i = 0; i < elements.size(); ++i) {
        const RealisticLensElement& element = elements[i];
        const float surfaceCenter = element.vertexZ + surfaceOffset;
        if (element.isAperture != 0) {
            if (std::abs(tracedRay.direction.z) <= 1e-7f)
                return false;
            const float t = (-surfaceCenter - tracedRay.origin.z) / tracedRay.direction.z;
            if (t < 0.0f)
                return false;
            const vec3 p = tracedRay.origin + tracedRay.direction * t;
            if (glm::length2(vec2(p.x, p.y)) > element.apertureRadius * element.apertureRadius)
                return false;
            continue;
        }

        float t = 0.0f;
        vec3 normal{};
        if (!intersectRossSphere(element, surfaceOffset, tracedRay, t, normal))
            return false;

        const vec3 point = tracedRay.origin + tracedRay.direction * t;
        if (glm::length2(vec2(point.x, point.y)) > element.apertureRadius * element.apertureRadius)
            return false;

        const float etaI = element.ior;
        const float etaT = (i > 0 && elements[i - 1].ior != 0.0f) ? elements[i - 1].ior : 1.0f;
        CpuRay refracted{};
        if (!refractRoss(point, normal, tracedRay, etaI, etaT, refracted))
            return false;
        tracedRay = {refracted.origin, -refracted.direction};
    }

    outputRay = {{tracedRay.origin.x, tracedRay.origin.y, -tracedRay.origin.z},
                 {tracedRay.direction.x, tracedRay.direction.y, -tracedRay.direction.z}};
    return true;
}

void computeCardinalPoints(const CpuRay& rayIn, const CpuRay& rayOut, float& pz, float& fz)
{
    const float tf = -rayOut.origin.x / rayOut.direction.x;
    fz = -(rayOut.origin + rayOut.direction * tf).z;
    const float tp = (rayIn.origin.x - rayOut.origin.x) / rayOut.direction.x;
    pz = -(rayOut.origin + rayOut.direction * tp).z;
}
}

RealisticCamera::RealisticCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings)
    : CameraBase(scene, name, transform, settings)
{
    loadLensAndSensor();
    rebuildCameraData();
}

RealisticCamera::RealisticCamera(Scene& scene, const std::string& name, Transform transform, CameraSettings settings, std::string lensPath, std::string sensorPath, std::string glassCatalogPaths)
    : CameraBase(scene, name, transform, settings),
      lensPath(std::move(lensPath)),
      sensorPath(std::move(sensorPath)),
      glassCatalogPaths(std::move(glassCatalogPaths))
{
    loadLensAndSensor();
    rebuildCameraData();
}

RealisticCamera::RealisticCamera(const RealisticCamera& other)
    : CameraBase(other),
      lensPath(other.lensPath),
      sensorPath(other.sensorPath),
      glassCatalogPaths(other.glassCatalogPaths),
      lensElements(other.lensElements),
      maximumLensElements(other.maximumLensElements),
      pupilBounds(other.pupilBounds),
      sensorWidthM(other.sensorWidthM),
      sensorHeightM(other.sensorHeightM),
      sensorResolutionWidth(other.sensorResolutionWidth),
      sensorResolutionHeight(other.sensorResolutionHeight),
      effectiveFocalLengthM(other.effectiveFocalLengthM),
      focusSurfaceOffsetM(other.focusSurfaceOffsetM),
      apertureIndex(other.apertureIndex),
      loadStatus(other.loadStatus)
{
    rebuildCameraData();
}

std::unique_ptr<SceneObject> RealisticCamera::clone() const
{
    return std::make_unique<RealisticCamera>(*this);
}

bool RealisticCamera::getPreferredRenderSize(uint32_t& width, uint32_t& height) const
{
    if (lensElements.empty() || sensorResolutionWidth == 0 || sensorResolutionHeight == 0)
        return false;
    width = sensorResolutionWidth;
    height = sensorResolutionHeight;
    return true;
}

mat4 RealisticCamera::getProjectionMatrix() const
{
    const float aspectRatio = static_cast<float>(std::max(1u, renderWidth)) / static_cast<float>(std::max(1u, renderHeight));
    const float fovY = 2.0f * std::atan(sensorHeightM / (2.0f * std::max(effectiveFocalLengthM, 0.001f)));
    return perspective(fovY, aspectRatio, cameraData.nearPlane, cameraData.farPlane);
}

void RealisticCamera::computeProjectionData(const vec3&, const vec3& up, const vec3& right, float aspectRatio)
{
    const float sensorWidth = std::max(sensorWidthM, 0.001f);
    const float focalLength = std::max(effectiveFocalLengthM, 0.001f);
    const float halfWidth = sensorWidth / (2.0f * focalLength);
    const float halfHeight = halfWidth / aspectRatio;

    cameraData.horizontal = right * (2.0f * halfWidth);
    cameraData.vertical = up * (2.0f * halfHeight);
    cameraData.focalLength = focalLength;
    cameraData.orthoHeight = 0.0f;
    cameraData.fisheyeFov = 0.0f;
    applyDepthOfField(effectiveFocalLengthM * 1000.0f);
}

void RealisticCamera::renderUi()
{
    SceneObject::renderUi();

    bool changed = false;
    ImGuiManager::tableRowLabel("Projection");
    static const CameraProjectionType projectionTypes[] = {
        CameraProjectionType::Perspective,
        CameraProjectionType::ThinLens,
        CameraProjectionType::Realistic,
        CameraProjectionType::Orthographic,
        CameraProjectionType::Fisheye,
    };
    static const char* projectionNames[] = {"Perspective", "Thin Lens", "Realistic", "Orthographic", "Fisheye"};
    constexpr int projectionCount = 5;
    int projectionIndex = 0;
    for (int i = 0; i < projectionCount; ++i)
        if (projectionTypes[i] == getProjectionType()) { projectionIndex = i; break; }
    if (ImGui::Combo("##CameraProjection", &projectionIndex, projectionNames, projectionCount)) {
        if (auto replacement = recreateAs(projectionTypes[projectionIndex]))
            scene.replaceObject(this, std::move(replacement));
        return;
    }

    if (lensDialog && lensDialog->ready(0)) {
        const auto selection = lensDialog->result();
        if (!selection.empty()) {
            lensPath = selection.front();
            changed = loadLensAndSensor();
        }
        lensDialog.reset();
    }
    if (sensorDialog && sensorDialog->ready(0)) {
        const auto selection = sensorDialog->result();
        if (!selection.empty()) {
            sensorPath = selection.front();
            changed = loadLensAndSensor();
        }
        sensorDialog.reset();
    }
    if (glassCatalogDialog && glassCatalogDialog->ready(0)) {
        const auto selection = glassCatalogDialog->result();
        if (!selection.empty()) {
            glassCatalogPaths = joinPaths(selection);
            changed = loadLensAndSensor();
        }
        glassCatalogDialog.reset();
    }

    std::array<char, 512> lensBuffer{};
    std::array<char, 512> sensorBuffer{};
    std::array<char, 1024> glassCatalogBuffer{};
    std::snprintf(lensBuffer.data(), lensBuffer.size(), "%s", lensPath.c_str());
    std::snprintf(sensorBuffer.data(), sensorBuffer.size(), "%s", sensorPath.c_str());
    std::snprintf(glassCatalogBuffer.data(), glassCatalogBuffer.size(), "%s", glassCatalogPaths.c_str());

    ImGuiManager::dragFloatRow("Sensor Width", sensorWidthM * 1000.0f, 0.01f, 0.001f, 500.0f, [&](float v) {
        sensorWidthM = std::max(v, 0.001f) * 0.001f;
        rebuildLensMetadata();
        applyAperture();
        applyFocus();
        rebuildExitPupilBounds();
        changed = true;
    });
    ImGuiManager::dragFloatRow("Sensor Height", sensorHeightM * 1000.0f, 0.01f, 0.001f, 500.0f, [&](float v) {
        sensorHeightM = std::max(v, 0.001f) * 0.001f;
        rebuildLensMetadata();
        applyAperture();
        applyFocus();
        rebuildExitPupilBounds();
        changed = true;
    });
    ImGuiManager::dragFloatRow("F-Stop", settings.fStop, 0.01f, 0.0f, 128.0f, [&](float v) {
        settings.fStop = std::max(0.0f, v);
        applyAperture();
        rebuildExitPupilBounds();
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focus Distance", settings.focusDistance, 0.01f, 0.001f, 10000.0f, [&](float v) {
        settings.focusDistance = std::max(0.001f, v);
        applyFocus();
        rebuildExitPupilBounds();
        changed = true;
    });

    ImGuiManager::tableRowLabel("Output Size");
    if (sensorResolutionWidth > 0 && sensorResolutionHeight > 0)
        ImGui::Text("%u x %u", sensorResolutionWidth, sensorResolutionHeight);
    else
        ImGui::TextUnformatted("No sensor resolution");

    ImGuiManager::tableRowLabel("Lens File");
    const float lensButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - lensButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticLensPath", lensBuffer.data(), lensBuffer.size())) {
        lensPath = lensBuffer.data();
        if (lensPath.empty())
            changed = loadLensAndSensor();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(lensDialog != nullptr);
    if (ImGui::Button("Select##RealisticLens", ImVec2(lensButtonWidth, 0))) {
        lensDialog = std::make_unique<pfd::open_file>(
            "Select Lens File",
            "/home/marcel/GitRepositories/ROSS/resources/lenses",
            std::vector<std::string>{"Lens Files", "*.olio *.zmx *.dat", "All Files", "*"});
    }
    ImGui::EndDisabled();

    ImGuiManager::tableRowLabel("Glass Catalogs");
    const float catalogButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - catalogButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticGlassCatalogPaths", glassCatalogBuffer.data(), glassCatalogBuffer.size())) {
        glassCatalogPaths = glassCatalogBuffer.data();
        if (glassCatalogPaths.empty())
            changed = loadLensAndSensor();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(glassCatalogDialog != nullptr);
    if (ImGui::Button("Select##RealisticGlassCatalogs", ImVec2(catalogButtonWidth, 0))) {
        glassCatalogDialog = std::make_unique<pfd::open_file>(
            "Select Glass Catalogs",
            "/home/marcel/GitRepositories/ROSS/resources/glasscatalogs",
            std::vector<std::string>{"Glass Catalogs", "*.agf *.AGF", "All Files", "*"},
            pfd::opt::multiselect);
    }
    ImGui::EndDisabled();

    ImGuiManager::tableRowLabel("Sensor File");
    const float sensorButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - sensorButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##RealisticSensorPath", sensorBuffer.data(), sensorBuffer.size())) {
        sensorPath = sensorBuffer.data();
        if (sensorPath.empty())
            changed = loadLensAndSensor();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(sensorDialog != nullptr);
    if (ImGui::Button("Select##RealisticSensor", ImVec2(sensorButtonWidth, 0))) {
        sensorDialog = std::make_unique<pfd::open_file>(
            "Select Sensor File",
            "/home/marcel/GitRepositories/ROSS/resources/sensors",
            std::vector<std::string>{"Sensor Files", "*.json", "All Files", "*"});
    }
    ImGui::EndDisabled();

    ImGuiManager::tableRowLabel("Optics");
    if (ImGui::Button("Reload##RealisticCamera")) {
        changed = loadLensAndSensor();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(loadStatus.c_str());

    ImGuiManager::tableRowLabel("Lens Elements");
    if (lensElements.empty()) {
        ImGui::TextUnformatted("Thin lens fallback");
    } else {
        constexpr float metersToMillimeters = 1000.0f;
        const float tableHeight = std::min(320.0f, 28.0f + static_cast<float>(lensElements.size()) * ImGui::GetTextLineHeightWithSpacing());
        if (ImGui::BeginChild("##RealisticLensElementsChild", ImVec2(0.0f, tableHeight), ImGuiChildFlags_Borders)) {
            constexpr ImGuiTableFlags tableFlags =
                ImGuiTableFlags_Borders |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_ScrollY |
                ImGuiTableFlags_SizingStretchProp;
            if (ImGui::BeginTable("##RealisticLensElementsTable", 7, tableFlags)) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 32.0f);
                ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                ImGui::TableSetupColumn("Radius mm");
                ImGui::TableSetupColumn("Thickness mm");
                ImGui::TableSetupColumn("Aperture mm");
                ImGui::TableSetupColumn("IOR");
                ImGui::TableSetupColumn("Z mm");
                ImGui::TableHeadersRow();

                for (size_t i = 0; i < lensElements.size(); ++i) {
                    const RealisticLensElement& element = lensElements[i];
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%zu", i);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextUnformatted(element.isAperture != 0 ? "Aperture" : "Surface");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.4g", element.radius * metersToMillimeters);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.4g", element.thickness * metersToMillimeters);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%.4g", element.apertureRadius * metersToMillimeters);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%.4g", element.ior);
                    ImGui::TableSetColumnIndex(6);
                    ImGui::Text("%.4g", element.vertexZ * metersToMillimeters);
                }

                ImGui::EndTable();
            }
        }
        ImGui::EndChild();
    }

    if (changed) {
        rebuildCameraData();
        scene.setDirtyFlag(Accumulation);
    }
}

void RealisticCamera::populateRealisticCameraSettings(RealisticCameraSettings& realisticSettings) const
{
    realisticSettings = {};
    const int count = std::min<int>(static_cast<int>(lensElements.size()), MAX_REALISTIC_LENS_ELEMENTS);
    for (int i = 0; i < count; ++i)
        realisticSettings.elements[i] = lensElements[static_cast<size_t>(i)];
    const int pupilCount = std::min<int>(static_cast<int>(pupilBounds.size()), MAX_REALISTIC_EXIT_PUPIL_BOUNDS);
    for (int i = 0; i < pupilCount; ++i)
        realisticSettings.pupilBounds[i] = pupilBounds[static_cast<size_t>(i)];
    realisticSettings.elementCount = count;
    realisticSettings.pupilBoundCount = pupilCount;
    realisticSettings.sensorWidth = sensorWidthM;
    realisticSettings.sensorHeight = sensorHeightM;
    realisticSettings.filmDiagonal = std::sqrt(sensorWidthM * sensorWidthM + sensorHeightM * sensorHeightM);
    realisticSettings.rearElementZ = count > 0 ? lensElements.back().vertexZ + focusSurfaceOffsetM : 0.0f;
    realisticSettings.apertureRadius = 0.0f;
    realisticSettings.surfaceOffset = focusSurfaceOffsetM;
    for (int i = 0; i < count; ++i) {
        const RealisticLensElement& element = lensElements[static_cast<size_t>(i)];
        if (element.isAperture != 0) {
            realisticSettings.apertureRadius = element.apertureRadius;
            break;
        }
        if (realisticSettings.apertureRadius <= 0.0f || element.apertureRadius < realisticSettings.apertureRadius)
            realisticSettings.apertureRadius = element.apertureRadius;
    }
}

bool RealisticCamera::loadLensAndSensor()
{
    if (lensPath.empty() || sensorPath.empty()) {
        lensElements.clear();
        maximumLensElements.clear();
        pupilBounds.clear();
        sensorResolutionWidth = 0;
        sensorResolutionHeight = 0;
        apertureIndex = -1;
        focusSurfaceOffsetM = 0.0f;
        loadStatus = "Thin lens fallback";
        std::cout << "[INFO] RealisticCamera: no lens/sensor file loaded, using thin lens fallback" << std::endl;
        return true;
    }

    std::string lensError;
    std::string sensorError;
    std::vector<RealisticLensElement> parsedElements;
    const bool lensOk = parseLensFile(lensPath, parsedElements, lensError);
    const bool sensorOk = parseSensorFile(sensorPath, sensorError);
    if (lensOk) {
        maximumLensElements = parsedElements;
        lensElements = std::move(parsedElements);
        rebuildLensMetadata();
        applyAperture();
        applyFocus();
    }

    if (lensOk && sensorOk) {
        rebuildExitPupilBounds();
        loadStatus = std::to_string(lensElements.size()) + " surfaces";
        std::cout << "[INFO] RealisticCamera: loaded " << lensElements.size() << " surfaces"
                  << ", pupil bounds: " << pupilBounds.size()
                  << ", effective focal length: " << effectiveFocalLengthM * 1000.0f << " mm" << std::endl;
        return true;
    }

    lensElements.clear();
    maximumLensElements.clear();
    pupilBounds.clear();
    apertureIndex = -1;
    focusSurfaceOffsetM = 0.0f;
    loadStatus = !lensOk ? lensError : sensorError;
    std::cerr << "[ERROR] RealisticCamera: " << loadStatus << std::endl;
    return false;
}

bool RealisticCamera::parseLensFile(const std::string& path, std::vector<RealisticLensElement>& elements, std::string& error) const
{
    const std::string text = readTextFile(path);
    if (text.empty()) {
        error = "Could not read lens";
        return false;
    }

    const std::string ext = lowerCopy(std::filesystem::path(path).extension().string());
    bool ok = false;
    if (ext == ".olio")
        ok = parseOlioFile(text, elements, error);
    else if (ext == ".zmx")
        ok = parseZmxFile(text, elements, error);
    else
        ok = parseDatFile(text, elements, error);

    if (ok)
        finalizeElements(elements);
    return ok;
}

bool RealisticCamera::parseOlioFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const
{
    bool inSurfaces = false;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        line = trimCopy(line);
        if (line == "[Surfaces]") {
            inSurfaces = true;
            continue;
        }
        if (inSurfaces && !line.empty() && line.front() == '[')
            break;
        if (!inSurfaces || line.empty() || std::isalpha(static_cast<unsigned char>(line.front())))
            continue;

        const auto tokens = splitTokens(line);
        if (tokens.size() < 4)
            continue;
        RealisticLensElement element{};
        if (!parseFloat(tokens[0], element.radius) ||
            !parseFloat(tokens[1], element.thickness) ||
            !parseFloat(tokens[2], element.ior) ||
            !parseFloat(tokens[3], element.apertureRadius))
            continue;
        element.isAperture = (std::abs(element.radius) < 1e-6f || element.ior <= 0.0f) ? 1 : 0;
        elements.push_back(element);
    }

    if (elements.empty()) {
        error = "No OLIO surfaces";
        return false;
    }
    return true;
}

bool RealisticCamera::parseDatFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const
{
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        line = trimCopy(line);
        if (line.empty() || line.front() == '#')
            continue;
        const auto tokens = splitTokens(line);
        if (tokens.size() < 4)
            continue;
        RealisticLensElement element{};
        if (!parseFloat(tokens[0], element.radius) ||
            !parseFloat(tokens[1], element.thickness) ||
            !parseFloat(tokens[2], element.ior) ||
            !parseFloat(tokens[3], element.apertureRadius))
            continue;
        element.isAperture = (std::abs(element.radius) < 1e-6f || element.ior <= 0.0f) ? 1 : 0;
        elements.push_back(element);
    }

    if (elements.empty()) {
        error = "No DAT surfaces";
        return false;
    }
    return true;
}

bool RealisticCamera::parseZmxFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const
{
    struct ZmxSurface {
        float curvature = 0.0f;
        float thickness = 0.0f;
        float ior = 1.0f;
        float diameter = 0.0f;
        std::string glassName;
        bool hasData = false;
        bool stop = false;
    };

    const auto glassCatalog = loadAgfCatalogs(glassCatalogPaths);
    std::vector<ZmxSurface> surfaces;
    ZmxSurface current;
    bool active = false;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        line = trimCopy(line);
        if (line.empty())
            continue;
        const auto tokens = splitTokens(line);
        if (tokens.empty())
            continue;
        const std::string key = tokens[0];
        if (key == "SURF") {
            if (active && current.hasData)
                surfaces.push_back(current);
            current = {};
            active = true;
            continue;
        }
        if (!active)
            continue;
        if (key == "STOP") {
            current.stop = true;
            current.hasData = true;
        } else if (key == "CURV" && tokens.size() > 1) {
            parseFloat(tokens[1], current.curvature);
            current.hasData = true;
        } else if (key == "DISZ" && tokens.size() > 1) {
            parseFloat(tokens[1], current.thickness);
            current.hasData = true;
        } else if (key == "GLAS") {
            if (tokens.size() > 1) {
                current.glassName = tokens[1];
                const auto catalogIt = glassCatalog.find(normalizeGlassName(current.glassName));
                if (catalogIt != glassCatalog.end()) {
                    current.ior = catalogIt->second;
                } else if (tokens.size() > 4) {
                    float explicitIor = 0.0f;
                    if (parseFloat(tokens[4], explicitIor) && explicitIor > 1.0f)
                        current.ior = explicitIor;
                }
            }
            current.hasData = true;
        } else if ((key == "DIAM" || key == "MEMA") && tokens.size() > 1) {
            parseFloat(tokens[1], current.diameter);
            current.hasData = true;
        }
    }
    if (active && current.hasData)
        surfaces.push_back(current);

    for (const ZmxSurface& surface : surfaces) {
        if (surface.diameter <= 0.0f)
            continue;
        if (std::abs(surface.curvature) < 1e-8f && surface.thickness == 0.0f && !surface.stop && std::abs(surface.ior - 1.0f) < 1e-6f)
            continue;
        RealisticLensElement element{};
        element.radius = std::abs(surface.curvature) > 1e-8f ? 1.0f / surface.curvature : 0.0f;
        element.thickness = surface.thickness;
        element.ior = surface.stop ? 0.0f : surface.ior;
        element.apertureRadius = surface.diameter * 0.5f;
        element.isAperture = (surface.stop || element.ior <= 0.0f) ? 1 : 0;
        elements.push_back(element);
    }


    if (elements.empty()) {
        error = "No ZMX surfaces";
        return false;
    }
    return true;
}

bool RealisticCamera::parseSensorFile(const std::string& path, std::string& error)
{
    sensorResolutionWidth = 0;
    sensorResolutionHeight = 0;

    const std::string text = readTextFile(path);
    if (text.empty()) {
        error = "Could not read sensor";
        return false;
    }

    float widthMm = 0.0f;
    float heightMm = 0.0f;
    if (!parseJsonNumber(text, "width_mm", widthMm) || !parseJsonNumber(text, "height_mm", heightMm)) {
        error = "Invalid sensor JSON";
        return false;
    }

    uint32_t resolutionWidth = 0;
    uint32_t resolutionHeight = 0;
    if (!parseJsonResolution(text, resolutionWidth, resolutionHeight)) {
        error = "Invalid sensor resolution";
        return false;
    }

    sensorWidthM = std::max(widthMm, 0.001f) * 0.001f;
    sensorHeightM = std::max(heightMm, 0.001f) * 0.001f;
    sensorResolutionWidth = resolutionWidth;
    sensorResolutionHeight = resolutionHeight;
    return true;
}

void RealisticCamera::finalizeElements(std::vector<RealisticLensElement>& elements) const
{
    if (elements.size() > MAX_REALISTIC_LENS_ELEMENTS)
        elements.resize(MAX_REALISTIC_LENS_ELEMENTS);

    float vertexZ = 0.0f;
    for (int i = static_cast<int>(elements.size()) - 1; i >= 0; --i) {
        RealisticLensElement& element = elements[static_cast<size_t>(i)];
        vertexZ += element.thickness;
        element.vertexZ = vertexZ * 0.001f;
        element.radius *= 0.001f;
        element.thickness *= 0.001f;
        element.apertureRadius *= 0.001f;
        if (element.ior <= 0.0f)
            element.isAperture = 1;
        if (element.isAperture)
            element.ior = 1.0f;
    }
}

void RealisticCamera::rebuildLensMetadata()
{
    apertureIndex = -1;
    for (int i = static_cast<int>(lensElements.size()) - 1; i >= 0; --i) {
        if (lensElements[static_cast<size_t>(i)].isAperture != 0) {
            apertureIndex = i;
            break;
        }
    }

    effectiveFocalLengthM = std::max(settings.focalLengthMm * 0.001f, 0.001f);
    if (lensElements.size() < 2)
        return;

    const float x = 0.001f * std::sqrt(sensorWidthM * sensorWidthM + sensorHeightM * sensorHeightM);
    CpuRay sceneRay{{x, 0.0f, lensElements.front().vertexZ + 1.0f}, {0.0f, 0.0f, -1.0f}};
    CpuRay filmRay{};
    CpuRay filmProbe{{x, 0.0f, lensElements.back().vertexZ - 1.0f}, {0.0f, 0.0f, 1.0f}};
    CpuRay sceneOut{};
    if (!traceFromSceneRoss(lensElements, 0.0f, sceneRay, filmRay))
        return;
    if (!traceFromFilmRoss(lensElements, 0.0f, filmProbe, sceneOut))
        return;

    std::cout << "[DEBUG FL] sceneRay O=(" << sceneRay.origin.x << "," << sceneRay.origin.y << "," << sceneRay.origin.z << ") D=(" << sceneRay.direction.x << "," << sceneRay.direction.y << "," << sceneRay.direction.z << ")" << std::endl;
    std::cout << "[DEBUG FL] filmRay O=(" << filmRay.origin.x << "," << filmRay.origin.y << "," << filmRay.origin.z << ") D=(" << filmRay.direction.x << "," << filmRay.direction.y << "," << filmRay.direction.z << ")" << std::endl;
    std::cout << "[DEBUG FL] filmProbe O=(" << filmProbe.origin.x << "," << filmProbe.origin.y << "," << filmProbe.origin.z << ") D=(" << filmProbe.direction.x << "," << filmProbe.direction.y << "," << filmProbe.direction.z << ")" << std::endl;
    std::cout << "[DEBUG FL] sceneOut O=(" << sceneOut.origin.x << "," << sceneOut.origin.y << "," << sceneOut.origin.z << ") D=(" << sceneOut.direction.x << "," << sceneOut.direction.y << "," << sceneOut.direction.z << ")" << std::endl;

    float pz0 = 0.0f;
    float fz0 = 0.0f;
    float pz1 = 0.0f;
    float fz1 = 0.0f;
    computeCardinalPoints(sceneRay, filmRay, pz0, fz0);
    std::cout << "[DEBUG FL] pz0=" << pz0 << " fz0=" << fz0 << std::endl;
    computeCardinalPoints(filmProbe, sceneOut, pz1, fz1);
    std::cout << "[DEBUG FL] pz1=" << pz1 << " fz1=" << fz1 << std::endl;
    const float focalLength = fz0 - pz0;
    if (std::isfinite(focalLength) && focalLength > 0.0f)
        effectiveFocalLengthM = focalLength;
}

void RealisticCamera::applyAperture()
{
    if (apertureIndex < 0 || apertureIndex >= static_cast<int>(lensElements.size()))
        return;

    const float maxRadius = apertureIndex < static_cast<int>(maximumLensElements.size())
        ? maximumLensElements[static_cast<size_t>(apertureIndex)].apertureRadius
        : lensElements[static_cast<size_t>(apertureIndex)].apertureRadius;
    const float requestedRadius = settings.fStop > 0.0f
        ? effectiveFocalLengthM / (2.0f * settings.fStop)
        : maxRadius;
    lensElements[static_cast<size_t>(apertureIndex)].apertureRadius = std::clamp(requestedRadius, 0.0f, maxRadius);
}

void RealisticCamera::applyFocus()
{
    focusSurfaceOffsetM = 0.0f;
    if (lensElements.size() < 2)
        return;

    const float x = 0.001f * std::sqrt(sensorWidthM * sensorWidthM + sensorHeightM * sensorHeightM);
    CpuRay sceneRay{{x, 0.0f, lensElements.front().vertexZ + 1.0f}, {0.0f, 0.0f, -1.0f}};
    CpuRay filmRay{};
    CpuRay filmProbe{{x, 0.0f, lensElements.back().vertexZ - 1.0f}, {0.0f, 0.0f, 1.0f}};
    CpuRay sceneOut{};
    if (!traceFromSceneRoss(lensElements, 0.0f, sceneRay, filmRay))
        return;
    if (!traceFromFilmRoss(lensElements, 0.0f, filmProbe, sceneOut))
        return;

    float pz0 = 0.0f;
    float fz0 = 0.0f;
    float pz1 = 0.0f;
    float fz1 = 0.0f;
    computeCardinalPoints(sceneRay, filmRay, pz0, fz0);
    computeCardinalPoints(filmProbe, sceneOut, pz1, fz1);

    const float f = fz0 - pz0;
    const float z = -settings.focusDistance;
    const float c = (pz1 - z - pz0) * (pz1 - z - 4.0f * f - pz0);
    if (c <= 0.0f)
        return;

    const float offset = 0.5f * (pz1 - z + pz0 - std::sqrt(c));
    if (std::isfinite(offset))
        focusSurfaceOffsetM = offset;
}

void RealisticCamera::rebuildExitPupilBounds()
{
    pupilBounds.clear();
    if (lensElements.empty())
        return;

    constexpr int boundsCount = MAX_REALISTIC_EXIT_PUPIL_BOUNDS;
    constexpr int samplesPerDimension = 48;
    const float filmDiagonal = std::sqrt(sensorWidthM * sensorWidthM + sensorHeightM * sensorHeightM);
    const float rearRadius = lensElements.back().apertureRadius;
    const float rearZ = lensElements.back().vertexZ + focusSurfaceOffsetM;
    if (filmDiagonal <= 0.0f || rearRadius <= 0.0f || rearZ <= 0.0f)
        return;

    pupilBounds.resize(boundsCount);
    const vec2 rearMin(-1.5f * rearRadius);
    const vec2 rearMax(1.5f * rearRadius);
    const float rearDiagonal = glm::length(rearMax - rearMin);
    const float expandAmount = 2.0f * rearDiagonal / static_cast<float>(samplesPerDimension);

    for (int interval = 0; interval < boundsCount; ++interval) {
        const float filmStart = (static_cast<float>(interval) / boundsCount) * (filmDiagonal * 0.5f);
        const float filmEnd = (static_cast<float>(interval + 1) / boundsCount) * (filmDiagonal * 0.5f);
        vec2 minBounds(std::numeric_limits<float>::max());
        vec2 maxBounds(-std::numeric_limits<float>::max());
        bool anyRayExited = false;

        for (int y = 0; y < samplesPerDimension; ++y) {
            for (int x = 0; x < samplesPerDimension; ++x) {
                const int sampleIndex = y * samplesPerDimension + x;
                const float filmT = (static_cast<float>(sampleIndex) + 0.5f) /
                    static_cast<float>(samplesPerDimension * samplesPerDimension);
                const float filmX = std::lerp(filmStart, filmEnd, filmT);
                const vec3 filmPoint(filmX, 0.0f, 0.0f);

                const vec2 u((static_cast<float>(x) + 0.5f) / samplesPerDimension,
                             (static_cast<float>(y) + 0.5f) / samplesPerDimension);
                const vec2 rearPoint = rearMin + (rearMax - rearMin) * u;
                if (rearPoint.x >= minBounds.x && rearPoint.y >= minBounds.y &&
                    rearPoint.x <= maxBounds.x && rearPoint.y <= maxBounds.y)
                    continue;

                CpuRay testRay{filmPoint, glm::normalize(vec3(rearPoint.x, rearPoint.y, rearZ) - filmPoint)};
                CpuRay tracedRay{};
                if (traceFromFilmRoss(lensElements, focusSurfaceOffsetM, testRay, tracedRay)) {
                    minBounds = glm::min(minBounds, rearPoint);
                    maxBounds = glm::max(maxBounds, rearPoint);
                    anyRayExited = true;
                }
            }
        }

        if (!anyRayExited) {
            pupilBounds[static_cast<size_t>(interval)] = {rearMin, rearMax};
            continue;
        }

        minBounds = glm::max(minBounds - vec2(expandAmount), rearMin);
        maxBounds = glm::min(maxBounds + vec2(expandAmount), rearMax);
        pupilBounds[static_cast<size_t>(interval)] = {minBounds, maxBounds};
    }
}
