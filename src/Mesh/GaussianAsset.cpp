#include "Mesh/GaussianAsset.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gf/core/gauss_ir.h>
#include <gf/io/ksplat.h>
#include <gf/io/ply_auto.h>
#include <gf/io/reader.h>
#include <gf/io/sog.h>
#include <gf/io/splat.h>
#include <gf/io/spz.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include "UI/ImGuiManager.h"
#include "Raytracing/RgbToSpectrum.h"
#include "Scene/CoordinateSystem.h"

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];

static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

namespace {
std::string lowercase(std::string value)
{
    std::ranges::transform(value, value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::unique_ptr<gf::IGaussReader> makeGaussianReaderForPath(const std::string& path)
{
    const std::filesystem::path filePath(path);
    const std::string filename = lowercase(filePath.filename().string());
    const std::string extension = lowercase(filePath.extension().string());

    if (extension == ".ply" || filename.ends_with(".compressed.ply"))
        return gf::MakePlyAutoReader();
    if (extension == ".splat")
        return gf::MakeSplatReader();
    if (extension == ".ksplat")
        return gf::MakeKsplatReader();
    if (extension == ".spz")
        return gf::MakeSpzReader();
    if (extension == ".sog")
        return gf::MakeSogReader();

    throw std::runtime_error("Unsupported Gaussian file format: " + path);
}

nr::coords::CoordinateSpace gaussianSourceSpace(const gf::GaussianCloudIR& ir)
{
    if (ir.meta.sourceFormat == "sog")
        return nr::coords::YDownZForwardSpace;
    if (ir.meta.handedness == gf::Handedness::kRight && ir.meta.up == gf::UpAxis::kY)
        return nr::coords::OpenGlSpace;
    if (ir.meta.handedness == gf::Handedness::kRight && ir.meta.up == gf::UpAxis::kZ)
        return nr::coords::ZUpYForwardSpace;

    // Raw 3DGS Gaussian files commonly come from COLMAP/OpenCV-style data:
    // x right, y down, z forward. Those files usually do not carry explicit
    // coordinate metadata, so convert that convention into NoorRay/OpenGL
    // space at asset import time instead of storing a corrective scene rotation.
    return nr::coords::YDownZForwardSpace;
}
}

GaussianAsset GaussianAsset::CreateFromFile(Scene& scene, const std::string& name, const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open Gaussian file: " + path);

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize <= 0)
        throw std::runtime_error("Gaussian file is empty: " + path);
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(fileSize));
    if (!file.read(reinterpret_cast<char*>(data.data()), fileSize))
        throw std::runtime_error("Failed to read Gaussian file: " + path);

    auto reader = makeGaussianReaderForPath(path);
    auto result = reader->Read(data.data(), data.size(), gf::ReadOptions{ .strict = true });
    if (!result)
        throw std::runtime_error("Failed to import Gaussian file: " + result.error().message);

    const gf::GaussianCloudIR& ir = result.value();
    const size_t count = static_cast<size_t>(ir.numPoints);
    const nr::coords::CoordinateSpace sourceSpace = gaussianSourceSpace(ir);

    static constexpr float SH_C0 = 0.28209479177387814f;

    nr::rstd::vector<Gaussian> gaussians;
    gaussians.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        Gaussian g;

        // Position
        const glm::vec3 position = nr::coords::toOpenGlVector({
            ir.positions[i * 3 + 0],
            ir.positions[i * 3 + 1],
            ir.positions[i * 3 + 2],
        }, sourceSpace);

        // Scale (log-space → linear). True per-axis sigma — the cutoff is
        // baked into the shared proxy geometry instead (GaussianCutoffSigma),
        // so this transform stays the untruncated R*S and is applied for
        // free by the hardware instance transform.
        const float sx = std::exp(ir.scales[i * 3 + 0]);
        const float sy = std::exp(ir.scales[i * 3 + 1]);
        const float sz = std::exp(ir.scales[i * 3 + 2]);

        // Rotation (wxyz order, normalize)
        glm::quat q;
        q.w = ir.rotations[i * 4 + 0];
        q.x = ir.rotations[i * 4 + 1];
        q.y = ir.rotations[i * 4 + 2];
        q.z = ir.rotations[i * 4 + 3];
        q = glm::normalize(q);

        // R*S: rotation from quat → mat3, then scale each column
        const glm::mat3 R = glm::mat3_cast(q);
        g.transform = glm::mat4x3(
            nr::coords::toOpenGlVector(R[0] * sx, sourceSpace),
            nr::coords::toOpenGlVector(R[1] * sy, sourceSpace),
            nr::coords::toOpenGlVector(R[2] * sz, sourceSpace),
            position
        );

        // SH degree-0 → RGB, clamped. Only feeds the spectral upsampling
        // below — the raw color is never stored on the Gaussian.
        float r = SH_C0 * ir.colors[i * 3 + 0] + 0.5f;
        float g_ = SH_C0 * ir.colors[i * 3 + 1] + 0.5f;
        float b = SH_C0 * ir.colors[i * 3 + 2] + 0.5f;
        r = std::max(r, 0.0f);
        g_ = std::max(g_, 0.0f);
        b = std::max(b, 0.0f);

        // Opacity: sigmoid of logit
        g.opacity = sigmoid(ir.alphas[i]);

        // Precompute the RGB->spectrum sigmoid-polynomial coefficients once,
        // here on the CPU, instead of doing this 64^3 table lookup per GPU hit.
        rgbLookupCoeffs(glm::vec3(r, g_, b), sRGBToSpectrumTable_Scale,
            reinterpret_cast<const float*>(sRGBToSpectrumTable_Data),
            g.spectrumCoeffs.x, g.spectrumCoeffs.y, g.spectrumCoeffs.z);

        gaussians.push_back(g);
    }

    GaussianAsset asset(scene, name, std::move(gaussians));
    asset.path = path;
    return asset;
}

GaussianAsset::GaussianAsset(Scene& scene, std::string name, nr::rstd::vector<Gaussian> gaussians)
    : scene(scene), name(std::move(name)), gaussians(std::move(gaussians)) {}

bool GaussianAsset::renderUi()
{
    ImGuiManager::tableRowLabel("Count");
    ImGui::Text("%zu", gaussians.size());

    ImGuiManager::tableRowLabel("Source");
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    ImGui::PushItemWidth(-1);
    ImGui::InputText("##gaussianPath", const_cast<char*>(path.c_str()), path.size() + 1, ImGuiInputTextFlags_ReadOnly);
    ImGui::PopItemWidth();
    ImGui::PopStyleColor();

    return false;
}
