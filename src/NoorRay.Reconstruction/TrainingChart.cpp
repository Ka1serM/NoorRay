#include "NoorRay.Reconstruction/TrainingChart.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Mesh.h>
#include <RmlUi/Core/RenderManager.h>

namespace NoorRay::Reconstruction {

namespace {

using Vec2 = Rml::Vector2f;

const Rml::Colourb gridColour{255, 255, 255, 18};
const Rml::Colourb lossColour{79, 163, 255, 255};
const Rml::Colourb spreadColour{79, 163, 255, 46};
const Rml::Colourb psnrColour{245, 165, 36, 255};
//! The width of one loss column in dp; the loss of the iterations in a column gives its mean and its spread.
constexpr float columnWidth = 2.0f;
constexpr float lineWidth = 1.5f;
constexpr float dotSize = 4.0f;
constexpr int gridLines = 4;

void addQuad(Rml::Mesh& mesh, const Vec2 a, const Vec2 b, const Vec2 c, const Vec2 d, const Rml::ColourbPremultiplied colour) {
    const int first = static_cast<int>(mesh.vertices.size());
    for (const Vec2 point : {a, b, c, d})
        mesh.vertices.push_back({point, colour, {0.0f, 0.0f}});
    mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

void addLine(Rml::Mesh& mesh, const Vec2 from, const Vec2 to, const float width, const Rml::ColourbPremultiplied colour) {
    const Vec2 along = to - from;
    const float length = along.Magnitude();
    if (length <= 0.0f)
        return;
    const Vec2 side = Vec2(-along.y, along.x) * (0.5f * width / length);
    addQuad(mesh, from + side, to + side, to - side, from - side, colour);
}

void addPolyline(Rml::Mesh& mesh, const std::vector<Vec2>& points, const float width, const Rml::ColourbPremultiplied colour) {
    for (std::size_t index = 1; index < points.size(); ++index)
        addLine(mesh, points[index - 1], points[index], width, colour);
}

ChartRange padded(float low, float high, const float minimum) {
    if (high - low < minimum) {
        const float middle = 0.5f * (low + high);
        low = middle - 0.5f * minimum;
        high = middle + 0.5f * minimum;
    }
    const float margin = 0.08f * (high - low);
    return {low - margin, high + margin};
}

struct Column {
    float low = std::numeric_limits<float>::max();
    float high = std::numeric_limits<float>::lowest();
    float sum = 0.0f;
    int count = 0;
};

}

void TrainingHistory::record(const TrainerStatus& status) {
    if (status.iteration <= 0)
        return;
    if (!lossIterations.empty() && static_cast<float>(status.iteration) < lossIterations.back())
        clear();
    planned = status.iterations;
    const auto iteration = static_cast<float>(status.iteration);
    if (!lossIterations.empty() && iteration == lossIterations.back())
        return;
    if (status.loss > 0.0) {
        lossIterations.push_back(iteration);
        logLoss.push_back(static_cast<float>(std::log10(status.loss)));
    }
    if (status.psnr > 0.0 && (psnr.empty() || static_cast<float>(status.psnr) != psnr.back())) {
        psnrIterations.push_back(iteration);
        psnr.push_back(static_cast<float>(status.psnr));
    }
    ++revision;
}

void TrainingHistory::clear() {
    lossIterations.clear();
    logLoss.clear();
    psnrIterations.clear();
    psnr.clear();
    ++revision;
}

float TrainingHistory::getIterationSpan() const {
    const float last = lossIterations.empty() ? 0.0f : lossIterations.back();
    return std::max({static_cast<float>(planned), last, 1.0f});
}

ChartRange TrainingHistory::getLossRange() const {
    if (logLoss.empty())
        return {-2.0f, 0.0f};
    const auto [low, high] = std::ranges::minmax(logLoss);
    return padded(low, high, 0.2f);
}

ChartRange TrainingHistory::getPsnrRange() const {
    if (psnr.empty())
        return {20.0f, 30.0f};
    const auto [low, high] = std::ranges::minmax(psnr);
    return padded(low, high, 1.0f);
}

TrainingChartElement::TrainingChartElement(const Rml::String& tag, const TrainingHistory& history) : Rml::Element(tag), history(history) {}

void TrainingChartElement::OnUpdate() {
    if (history.getRevision() != builtRevision)
        built = false;
}

void TrainingChartElement::OnRender() {
    if (!built)
        build();
    geometry.Render(GetAbsoluteOffset(Rml::BoxArea::Content));
}

void TrainingChartElement::build() {
    built = true;
    builtRevision = history.getRevision();
    Rml::RenderManager* renderManager = GetRenderManager();
    if (!renderManager)
        return;
    const Vec2 size = GetBox().GetSize(Rml::BoxArea::Content);
    const float density = GetContext()->GetDensityIndependentPixelRatio();
    Rml::Mesh mesh;
    for (int line = 0; line < gridLines; ++line) {
        const float y = std::round(size.y * static_cast<float>(line) / static_cast<float>(gridLines - 1));
        addQuad(mesh, {0.0f, y - 0.5f * density}, {size.x, y - 0.5f * density}, {size.x, y + 0.5f * density}, {0.0f, y + 0.5f * density},
                gridColour.ToPremultiplied());
    }
    const float span = history.getIterationSpan();
    const auto xOf = [&](const float iteration) { return size.x * iteration / span; };
    const auto yOf = [&](const float value, const ChartRange range) { return size.y * (1.0f - (value - range.low) / (range.high - range.low)); };

    const ChartRange lossRange = history.getLossRange();
    const std::vector<float>& iterations = history.getLossIterations();
    const std::vector<float>& logLoss = history.getLogLoss();
    const auto columnCount = static_cast<std::size_t>(std::max(1.0f, std::ceil(size.x / (columnWidth * density))));
    std::vector<Column> columns(columnCount);
    for (std::size_t index = 0; index < iterations.size(); ++index) {
        const auto column = std::min(columnCount - 1, static_cast<std::size_t>(xOf(iterations[index]) / (columnWidth * density)));
        Column& target = columns[column];
        target.low = std::min(target.low, logLoss[index]);
        target.high = std::max(target.high, logLoss[index]);
        target.sum += logLoss[index];
        ++target.count;
    }
    std::vector<Vec2> means;
    const Column* previous = nullptr;
    float previousX = 0.0f;
    for (std::size_t index = 0; index < columnCount; ++index) {
        const Column& column = columns[index];
        if (column.count == 0)
            continue;
        const float x = (static_cast<float>(index) + 0.5f) * columnWidth * density;
        if (previous)
            addQuad(mesh, {previousX, yOf(previous->high, lossRange)}, {x, yOf(column.high, lossRange)}, {x, yOf(column.low, lossRange)},
                    {previousX, yOf(previous->low, lossRange)}, spreadColour.ToPremultiplied());
        means.push_back({x, yOf(column.sum / static_cast<float>(column.count), lossRange)});
        previous = &column;
        previousX = x;
    }
    addPolyline(mesh, means, lineWidth * density, lossColour.ToPremultiplied());

    const ChartRange psnrRange = history.getPsnrRange();
    std::vector<Vec2> psnrPoints;
    for (std::size_t index = 0; index < history.getPsnr().size(); ++index)
        psnrPoints.push_back({xOf(history.getPsnrIterations()[index]), yOf(history.getPsnr()[index], psnrRange)});
    addPolyline(mesh, psnrPoints, lineWidth * density, psnrColour.ToPremultiplied());
    const float half = 0.5f * dotSize * density;
    for (const Vec2 point : psnrPoints)
        addQuad(mesh, point + Vec2(-half, -half), point + Vec2(half, -half), point + Vec2(half, half), point + Vec2(-half, half), psnrColour.ToPremultiplied());

    geometry = mesh.indices.empty() ? Rml::Geometry() : renderManager->MakeGeometry(std::move(mesh));
}

Rml::ElementPtr TrainingChartInstancer::InstanceElement(Rml::Element*, const Rml::String& tag, const Rml::XMLAttributes&) {
    return Rml::ElementPtr(new TrainingChartElement(tag, history));
}

void TrainingChartInstancer::ReleaseElement(Rml::Element* element) {
    delete element;
}

}
