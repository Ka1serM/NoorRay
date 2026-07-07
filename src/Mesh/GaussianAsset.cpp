#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include "Mesh/GaussianAsset.h"

#include <fstream>
#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include "UI/ImGuiManager.h"
#include "Raytracing/RgbToSpectrum.h"

extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];

static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

GaussianAsset GaussianAsset::CreateFromPly(Scene& scene, const std::string& name, const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open PLY file: " + path);

    tinyply::PlyFile ply;
    ply.parse_header(file);

    std::shared_ptr<tinyply::PlyData> xyz, opacity, scales, quat_rot, f_dc;

    try { xyz = ply.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception&) { throw std::runtime_error("PLY missing vertex x/y/z"); }

    try { opacity = ply.request_properties_from_element("vertex", { "opacity" }); }
    catch (const std::exception&) { throw std::runtime_error("PLY missing vertex opacity"); }

    try { scales = ply.request_properties_from_element("vertex", { "scale_0", "scale_1", "scale_2" }); }
    catch (const std::exception&) { throw std::runtime_error("PLY missing vertex scale_0/1/2"); }

    try { quat_rot = ply.request_properties_from_element("vertex", { "rot_0", "rot_1", "rot_2", "rot_3" }); }
    catch (const std::exception&) { throw std::runtime_error("PLY missing vertex rot_0/1/2/3"); }

    // SH DC coefficients (degree 0)
    try { f_dc = ply.request_properties_from_element("vertex", { "f_dc_0", "f_dc_1", "f_dc_2" }); }
    catch (const std::exception&) { throw std::runtime_error("PLY missing vertex f_dc_0/1/2"); }

    // f_rest_* (higher SH bands) are ignored in v1 — parse header but don't request data
    // so tinyply skips them during the read pass.

    ply.read(file);

    const size_t count = xyz->count;
    const float* xyzData  = reinterpret_cast<const float*>(xyz->buffer.get());
    const float* opData   = reinterpret_cast<const float*>(opacity->buffer.get());
    const float* scaleData = reinterpret_cast<const float*>(scales->buffer.get());
    const float* rotData  = reinterpret_cast<const float*>(quat_rot->buffer.get());
    const float* shData   = reinterpret_cast<const float*>(f_dc->buffer.get());

    static constexpr float SH_C0 = 0.28209479177387814f;

    nr::rstd::vector<Gaussian> gaussians;
    gaussians.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        Gaussian g;

        // Position
        const float px = xyzData[i * 3 + 0];
        const float py = xyzData[i * 3 + 1];
        const float pz = xyzData[i * 3 + 2];

        // Scale (log-space → linear). True per-axis sigma — the cutoff is
        // baked into the shared proxy geometry instead (GaussianCutoffSigma),
        // so this transform stays the untruncated R*S and is applied for
        // free by the hardware instance transform.
        const float sx = std::exp(scaleData[i * 3 + 0]);
        const float sy = std::exp(scaleData[i * 3 + 1]);
        const float sz = std::exp(scaleData[i * 3 + 2]);

        // Rotation (wxyz order, normalize)
        glm::quat q;
        q.w = rotData[i * 4 + 0];
        q.x = rotData[i * 4 + 1];
        q.y = rotData[i * 4 + 2];
        q.z = rotData[i * 4 + 3];
        q = glm::normalize(q);

        // R*S: rotation from quat → mat3, then scale each column
        const glm::mat3 R = glm::mat3_cast(q);
        g.transform = glm::mat4x3(
            R[0] * sx,
            R[1] * sy,
            R[2] * sz,
            glm::vec3(px, py, pz)
        );

        // SH degree-0 → RGB, clamped. Only feeds the spectral upsampling
        // below — the raw color is never stored on the Gaussian.
        float r = SH_C0 * shData[i * 3 + 0] + 0.5f;
        float g_ = SH_C0 * shData[i * 3 + 1] + 0.5f;
        float b = SH_C0 * shData[i * 3 + 2] + 0.5f;
        r = std::max(r, 0.0f);
        g_ = std::max(g_, 0.0f);
        b = std::max(b, 0.0f);

        // Opacity: sigmoid of logit
        g.opacity = sigmoid(opData[i]);

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
