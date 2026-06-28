#pragma once

#include "GPU/Annotations.h"
#include "Camera/Camera.h"
#include "Kernels/Samplers.h"
#include "Scene/SceneTypes.h"

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

NR_CPU_GPU inline bool refractLens(glm::vec3 pt, glm::vec3 normalIn, glm::vec3 rd,
    float iorIn, float iorOut, glm::vec3& outOrigin, glm::vec3& outDir)
{
    outOrigin = pt; outDir = rd;
    glm::vec3 nn = -normalIn;
    float cosI = glm::dot(nn, rd);
    float eta = iorIn / std::max(iorOut, 1e-7f);
    if (cosI < 0.0f) { eta = 1.0f / std::max(eta, 1e-7f); cosI = -cosI; nn = -nn; }
    const float sin2I = std::max(0.0f, 1.0f - cosI * cosI);
    const float sin2T = sin2I / std::max(eta * eta, 1e-7f);
    if (sin2T >= 1.0f) return false;
    const float cosT = std::sqrt(1.0f - sin2T);
    outDir = glm::normalize(-rd / eta + nn * (cosI / eta - cosT));
    return true;
}

NR_CPU_GPU inline bool traceFromFilm(
    const RealisticLensElement* elements, int elementCount,
    float surfaceOffset,
    glm::vec3 ro, glm::vec3 rd,
    glm::vec3& outOrigin, glm::vec3& outDir)
{
    outOrigin = ro; outDir = rd;
    if (elementCount <= 0) return false;

    glm::vec3 tracedO = ro;
    glm::vec3 tracedD = rd;

    for (int i = elementCount - 1; i >= 0; --i)
    {
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

        const float etaI = elem.ior;
        const float etaT = (i > 0 && elements[i - 1].ior != 0.0f)
            ? elements[i - 1].ior : 1.0f;

        glm::vec3 newO, newD;
        if (!refractLens(pt, n, tracedD, etaT, etaI, newO, newD)) return false;
        tracedO = newO;
        tracedD = -newD;
    }

    outOrigin = tracedO;
    outDir    = tracedD;
    return true;
}

NR_CPU_GPU inline glm::vec2 sampleExitPupil(
    const RealisticPupilBound* exitPupilBounds,
    int pupilBoundCount,
    float filmDiagonal,
    glm::vec2 filmPos,
    glm::vec2 u)
{
    const float rFilm = glm::length(filmPos);
    const float halfDiag = std::max(filmDiagonal * 0.5f, 1e-7f);
    int bi = static_cast<int>((rFilm / halfDiag) * static_cast<float>(pupilBoundCount));
    bi = glm::clamp(bi, 0, std::max(pupilBoundCount - 1, 0));
    const RealisticPupilBound& bound = exitPupilBounds[bi];
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

class CameraInstance;

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

    NR_CPU_GPU bool generateRay(glm::vec3& origin, glm::vec3& direction,
        float nx, float ny, uint32_t& rng, uint32_t) const
    {
        if (elementCount <= 0 || pupilBoundCount <= 0)
            return false;

        const glm::vec2 filmPos(-nx * sensorWidth * 0.5f, -ny * sensorHeight * 0.5f);

        glm::vec2 u(0.5f);
        if (fStop > 0.0f)
            u = glm::vec2(randomFloat(rng), randomFloat(rng));

        const glm::vec2 pupilPos = sampleExitPupil(
            exitPupilBounds, pupilBoundCount, filmDiagonal, filmPos, u);

        const glm::vec3 ro(filmPos.x, filmPos.y, 0.0f);
        const glm::vec3 filmDir = glm::normalize(
            glm::vec3(pupilPos.x - filmPos.x, pupilPos.y - filmPos.y, rearElementZ));

        glm::vec3 outO, outD;
        if (!traceFromFilm(elements, elementCount, surfaceOffset, ro, filmDir, outO, outD))
            return false;

        origin = outO;
        direction = glm::normalize(outD);
        transformRay(origin, direction);
        return true;
    }

#ifndef NR_GPU_CODE
    ~RealisticCamera();
    void renderUi(CameraInstance& inst);
    void load(std::string lensPath, std::string sensorPath, std::string glassCatalogPaths);

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
