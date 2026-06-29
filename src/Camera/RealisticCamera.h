#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Kernels/Samplers.h"
#include <glm/vec2.hpp>
using glm::vec2;

static constexpr int MaxRealisticLensElements = 64;
static constexpr int MaxRealisticExitPupilBounds = 64;

struct RealisticLensElement
{
    float radius{};
    float thickness{};
    float apertureRadius{};
    float ior{};
    float vertexZ{};
    int isAperture{};
    int _pad0{};
    int _pad1{};
};

struct RealisticPupilBound
{
    vec2 minBounds{};
    vec2 maxBounds{};
    float pupilArea{};
    float _pad0{};
};

NR_CPU_GPU inline bool intersectLensElement(
    const RealisticLensElement& elem, float surfaceOffset,
    glm::vec3 ro, glm::vec3 rd, float& t, glm::vec3& n)
{
    t = 0.0f; n = glm::vec3(0.0f, 0.0f, 1.0f);

    const float centerZ = elem.vertexZ + surfaceOffset;

    if (std::abs(elem.radius) <= 1e-7f)
    {
        if (std::abs(rd.z) <= 1e-7f) return false;
        t = (centerZ - ro.z) / rd.z;
        n = rd.z > 0.0f ? glm::vec3(0.0f, 0.0f, -1.0f) : glm::vec3(0.0f, 0.0f, 1.0f);
        return t >= 0.0f;
    }

    const float zCenter = centerZ + elem.radius;
    const glm::vec3 o = ro - glm::vec3(0.0f, 0.0f, zCenter);
    const float a = glm::dot(rd, rd);
    const float b = 2.0f * glm::dot(rd, o);
    const float c = glm::dot(o, o) - elem.radius * elem.radius;
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return false;

    const float root = std::sqrt(disc);
    const float t0 = (-b - root) / (2.0f * a);
    const float t1 = (-b + root) / (2.0f * a);
    const bool useSmaller = (rd.z > 0.0f) != (elem.radius < 0.0f);
    t = useSmaller ? std::min(t0, t1) : std::max(t0, t1);
    if (t < 0.0f) return false;

    n = glm::normalize(o + rd * t);
    if (glm::dot(n, -rd) < 0.0f) n = -n;
    return true;
}

NR_CPU_GPU inline bool refractLens(
    const glm::vec3 incident, glm::vec3 normal,
    const float etaIncident, const float etaTransmitted, glm::vec3& transmitted)
{
    if (glm::dot(normal, incident) > 0.0f)
        normal = -normal;
    const float cosIncident = fmaxf(-glm::dot(normal, incident), 0.0f);
    const float eta = etaIncident / fmaxf(etaTransmitted, 1e-7f);
    const float sin2Transmitted = eta * eta * fmaxf(0.0f, 1.0f - cosIncident * cosIncident);
    if (sin2Transmitted >= 1.0f)
        return false;
    const float cosTransmitted = sqrtf(1.0f - sin2Transmitted);
    transmitted = glm::normalize(
        eta * incident + (eta * cosIncident - cosTransmitted) * normal);
    return true;
}

NR_CPU_GPU inline bool traceLensSystem(
    const RealisticLensElement* elements, int elementCount,
    float surfaceOffset, bool fromFilm,
    glm::vec3 ro, glm::vec3 rd,
    glm::vec3& outOrigin, glm::vec3& outDir)
{
    outOrigin = ro; outDir = rd;
    if (elementCount <= 0) return false;

    glm::vec3 tracedO = ro;
    glm::vec3 tracedD = rd;

    for (int step = 0; step < elementCount; ++step)
    {
        const int i = fromFilm ? elementCount - 1 - step : step;
        const RealisticLensElement& elem = elements[i];
        const float surfaceCenter = elem.vertexZ + surfaceOffset;

        if (elem.isAperture != 0)
        {
            if (std::abs(tracedD.z) <= 1e-7f) return false;
            const float t = (surfaceCenter - tracedO.z) / tracedD.z;
            if (t < 0.0f) return false;
            const glm::vec3 p = tracedO + tracedD * t;
            if (p.x * p.x + p.y * p.y > elem.apertureRadius * elem.apertureRadius) return false;
            continue;
        }

        float t; glm::vec3 n;
        if (!intersectLensElement(elem, surfaceOffset, tracedO, tracedD, t, n)) return false;
        const glm::vec3 pt = tracedO + tracedD * t;
        if (pt.x * pt.x + pt.y * pt.y > elem.apertureRadius * elem.apertureRadius) return false;

        const float etaIncident = fromFilm
            ? fmaxf(elem.ior, 1.0f)
            : (i == 0 ? 1.0f : fmaxf(elements[i - 1].ior, 1.0f));
        const float etaTransmitted = fromFilm
            ? (i == 0 ? 1.0f : fmaxf(elements[i - 1].ior, 1.0f))
            : fmaxf(elem.ior, 1.0f);

        glm::vec3 refracted{};
        if (!refractLens(tracedD, n, etaIncident, etaTransmitted, refracted))
            return false;
        tracedO = pt;
        tracedD = refracted;
    }

    outOrigin = tracedO;
    outDir    = tracedD;
    return true;
}

NR_CPU_GPU inline bool traceFromFilm(
    const RealisticLensElement* elements, const int elementCount,
    const float surfaceOffset, const glm::vec3 ro, const glm::vec3 rd,
    glm::vec3& outOrigin, glm::vec3& outDir)
{
    return traceLensSystem(elements, elementCount, surfaceOffset, true,
        ro, rd, outOrigin, outDir);
}

NR_CPU_GPU inline glm::vec2 sampleExitPupil(
    const RealisticPupilBound* exitPupilBounds,
    int pupilBoundCount,
    float filmDiagonal,
    glm::vec2 filmPos,
    glm::vec2 u, float& pupilArea)
{
    const float rFilm = glm::length(filmPos);
    const float halfDiag = std::max(filmDiagonal * 0.5f, 1e-7f);
    int bi = static_cast<int>((rFilm / halfDiag) * static_cast<float>(pupilBoundCount));
    bi = glm::clamp(bi, 0, std::max(pupilBoundCount - 1, 0));
    const RealisticPupilBound& bound = exitPupilBounds[bi];
    pupilArea = bound.pupilArea;
    const glm::vec2 p = bound.minBounds + (bound.maxBounds - bound.minBounds) * u;
    const float sinTh = rFilm > 0.0f ? filmPos.y / rFilm : 0.0f;
    const float cosTh = rFilm > 0.0f ? filmPos.x / rFilm : 1.0f;
    return glm::vec2(cosTh * p.x - sinTh * p.y, sinTh * p.x + cosTh * p.y);
}

#ifndef NR_GPU_CODE
#include <memory>
#include <string>
#include <vector>
#include "portable-file-dialogs.h"
#endif

class RealisticCamera : public Camera {
public:
    RealisticLensElement elements[MaxRealisticLensElements]{};
    RealisticPupilBound exitPupilBounds[MaxRealisticExitPupilBounds]{};
    int elementCount{};
    int pupilBoundCount{};
    float sensorWidth{};
    float sensorHeight{};
    float apertureRadius{};
    float filmDiagonal{};
    float rearElementZ{};
    float surfaceOffset{};
    float onAxisPupilArea{};

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction, float& weight,
        float nx, float ny, uint32_t& rng, uint32_t) const
    {
        weight = 0.0f;
        if (elementCount <= 0 || pupilBoundCount <= 0)
            return false;

        const glm::vec2 filmPos(-nx * sensorWidth * 0.5f, -ny * sensorHeight * 0.5f);

        const glm::vec3 ro(filmPos.x, filmPos.y, 0.0f);
        glm::vec3 outO{}, outD{};
        float pupilArea = 0.0f;
        glm::vec3 filmDir{};
        bool traced = false;
        for (int attempt = 0; attempt < 16 && !traced; ++attempt)
        {
            const glm::vec2 pupilPos = sampleExitPupil(
                exitPupilBounds, pupilBoundCount, filmDiagonal, filmPos,
                glm::vec2(randomFloat(rng), randomFloat(rng)), pupilArea);
            filmDir = glm::normalize(glm::vec3(
                pupilPos.x - filmPos.x, pupilPos.y - filmPos.y, rearElementZ));
            traced = traceFromFilm(
                elements, elementCount, surfaceOffset, ro, filmDir, outO, outD);
        }
        if (!traced)
            return false;

        const float cosTheta = fabsf(filmDir.z);
        weight = pupilArea > 0.0f && onAxisPupilArea > 0.0f
            ? (pupilArea / onAxisPupilArea) * cosTheta * cosTheta * cosTheta * cosTheta
            : 1.0f;

        origin = outO;
        direction = glm::normalize(outD);
        transformRay(origin, direction);
        return true;
    }

#ifndef NR_GPU_CODE
    ~RealisticCamera();
    bool renderUi();
    void load(std::string lensPath, std::string sensorPath, std::string glassCatalogPaths);
    float derivedFocalLengthMm() const { return effectiveFocalLengthM * 1000.0f; }

private:
    std::string lensPath;
    std::string sensorPath;
    std::string glassCatalogPaths;
    std::vector<RealisticLensElement> lensElements;
    std::vector<RealisticLensElement> maximumLensElements;
    std::vector<RealisticPupilBound> pupilBounds;
    float effectiveFocalLengthM = 0.045f;
    float focusSurfaceOffsetM = 0.f;
    float firstPrincipalZ{};
    float secondPrincipalZ{};
    bool thickLensValid{};
    int apertureIndex = -1;
    std::string loadStatus;
    std::unique_ptr<pfd::open_file> lensDialog;
    std::unique_ptr<pfd::open_file> sensorDialog;
    std::unique_ptr<pfd::open_file> glassCatalogDialog;

    bool loadLensAndSensor();
    bool parseLensFile(const std::string& path, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseOlioFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseDatFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseZmxFile(const std::string& text, std::vector<RealisticLensElement>& elements, std::string& error) const;
    bool parseSensorFile(const std::string& path, std::string& error);
    void finalizeElements(std::vector<RealisticLensElement>& elements) const;
    void rebuildExitPupilBounds();
    void rebuildLensMetadata();
    void applyAperture();
    void applyFocus();
    void updateGpuData();
#endif
};
