#include "NoorRay/Materials/Nodes/Folder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <string_view>

namespace NoorRay::Materials {

namespace {

constexpr std::size_t side = foldSide;
constexpr std::size_t count = side * side;

using Samples = std::vector<Core::float4>;

Samples constant(const Core::float4 value) {
    return Samples(count, value);
}

Samples splat(const std::vector<float>& values) {
    Samples result(count);
    for (std::size_t index = 0; index < count; ++index)
        result[index] = {values[index], values[index], values[index], 1.0f};
    return result;
}

std::vector<float> component(const Samples& values, const std::size_t channel) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index)
        result[index] = (&values[index].x)[channel];
    return result;
}

float luminance(const Core::float4 value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

std::vector<float> luminances(const Samples& values) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index)
        result[index] = luminance(values[index]);
    return result;
}

Samples toScalar(const std::string_view type, const Samples& values) {
    if (type == "RGBA")
        return splat(luminances(values));
    if (type == "VECTOR")
        return splat(component(values, 0));
    return values;
}

Core::float4 lerp(const Core::float4 a, const Core::float4 b, const float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
}

template <class Operation>
Core::float4 apply(const Core::float4 a, const Core::float4 b, Operation operation) {
    return {operation(a.x, b.x), operation(a.y, b.y), operation(a.z, b.z), operation(a.w, b.w)};
}

Core::float4 clamp01(const Core::float4 value) {
    return apply(value, value, [](const float x, float) { return std::clamp(x, 0.0f, 1.0f); });
}

Core::float4 blend(const std::string_view mode, const float factor, const Core::float4 a, const Core::float4 b) {
    if (mode == "SCREEN")
        return apply(a, b, [&](const float x, const float y) { return 1.0f - (1.0f - factor + factor * (1.0f - y)) * (1.0f - x); });
    Core::float4 mixed = b;
    if (mode == "MULTIPLY")
        mixed = apply(a, b, std::multiplies<>());
    else if (mode == "ADD")
        mixed = apply(a, b, std::plus<>());
    else if (mode == "SUBTRACT")
        mixed = apply(a, b, std::minus<>());
    else if (mode == "DIVIDE")
        mixed = apply(a, b, [](const float x, const float y) { return y != 0.0f ? x / y : x; });
    else if (mode == "DIFFERENCE")
        mixed = apply(a, b, [](const float x, const float y) { return std::abs(x - y); });
    else if (mode == "DARKEN")
        mixed = apply(a, b, [](const float x, const float y) { return std::min(x, y); });
    else if (mode == "LIGHTEN")
        mixed = apply(a, b, [](const float x, const float y) { return std::max(x, y); });
    else if (mode == "OVERLAY")
        mixed = apply(a, b, [](const float x, const float y) { return x < 0.5f ? 2.0f * x * y : 1.0f - 2.0f * (1.0f - x) * (1.0f - y); });
    return lerp(a, mixed, factor);
}

Core::float3 rgbToHsv(const Core::float3 rgb) {
    const float high = std::max({rgb.x, rgb.y, rgb.z});
    const float low = std::min({rgb.x, rgb.y, rgb.z});
    if (high == low)
        return {0.0f, 0.0f, high};
    const float range = high - low;
    const float red = (high - rgb.x) / range;
    const float green = (high - rgb.y) / range;
    const float blue = (high - rgb.z) / range;
    float hue = rgb.x == high ? blue - green : rgb.y == high ? 2.0f + red - blue : 4.0f + green - red;
    hue = hue / 6.0f - std::floor(hue / 6.0f);
    return {hue, range / high, high};
}

Core::float3 hsvToRgb(const Core::float3 hsv) {
    const float value = hsv.z;
    if (hsv.y == 0.0f)
        return {value, value, value};
    const float scaled = hsv.x * 6.0f;
    const int sector = static_cast<int>(std::floor(scaled));
    const float fraction = scaled - static_cast<float>(sector);
    const float p = value * (1.0f - hsv.y);
    const float q = value * (1.0f - hsv.y * fraction);
    const float t = value * (1.0f - hsv.y * (1.0f - fraction));
    switch (((sector % 6) + 6) % 6) {
    case 0:
        return {value, t, p};
    case 1:
        return {q, value, p};
    case 2:
        return {p, value, t};
    case 3:
        return {p, q, value};
    case 4:
        return {t, p, value};
    default:
        return {value, p, q};
    }
}

std::optional<std::function<float(float, float, float)>> mathOperation(const std::string_view name) {
    if (name == "ADD")
        return [](const float a, const float b, float) { return a + b; };
    if (name == "SUBTRACT")
        return [](const float a, const float b, float) { return a - b; };
    if (name == "MULTIPLY")
        return [](const float a, const float b, float) { return a * b; };
    if (name == "DIVIDE")
        return [](const float a, const float b, float) { return b != 0.0f ? a / b : 0.0f; };
    if (name == "MULTIPLY_ADD")
        return [](const float a, const float b, const float c) { return a * b + c; };
    if (name == "POWER")
        return [](const float a, const float b, float) { return a > 0.0f ? std::pow(a, b) : 0.0f; };
    if (name == "MINIMUM")
        return [](const float a, const float b, float) { return std::min(a, b); };
    if (name == "MAXIMUM")
        return [](const float a, const float b, float) { return std::max(a, b); };
    if (name == "ABSOLUTE")
        return [](const float a, float, float) { return std::abs(a); };
    if (name == "SQRT")
        return [](const float a, float, float) { return std::sqrt(std::max(a, 0.0f)); };
    if (name == "LESS_THAN")
        return [](const float a, const float b, float) { return a < b ? 1.0f : 0.0f; };
    if (name == "GREATER_THAN")
        return [](const float a, const float b, float) { return a > b ? 1.0f : 0.0f; };
    if (name == "FLOOR")
        return [](const float a, float, float) { return std::floor(a); };
    if (name == "CEIL")
        return [](const float a, float, float) { return std::ceil(a); };
    if (name == "FRACT")
        return [](const float a, float, float) { return a - std::floor(a); };
    if (name == "SINE")
        return [](const float a, float, float) { return std::sin(a); };
    if (name == "COSINE")
        return [](const float a, float, float) { return std::cos(a); };
    if (name == "EXPONENT")
        return [](const float a, float, float) { return std::exp(a); };
    return std::nullopt;
}

bool isTexture(const std::string_view kind) {
    constexpr std::string_view textures[] = {"TEX_NOISE", "TEX_MUSGRAVE", "TEX_VORONOI", "TEX_WAVE",       "TEX_GRADIENT",
                                             "TEX_MAGIC", "TEX_CHECKER",  "TEX_BRICK",   "TEX_WHITE_NOISE"};
    return std::ranges::find(textures, kind) != std::end(textures);
}

bool isCoordinate(const std::string_view kind) {
    constexpr std::string_view coordinates[] = {"TEX_COORD", "UVMAP", "MAPPING", "NEW_GEOMETRY", "ATTRIBUTE", "VECT_TRANSFORM", "NORMAL_MAP", "BUMP"};
    return std::ranges::find(coordinates, kind) != std::end(coordinates);
}

}

Folder::Samples Folder::evaluate(const SocketRef socket, const std::size_t components) {
    links = NodeLinks();
    unknown = false;
    gridName.clear();
    Samples values = input(socket);
    if (unknown)
        return values;
    if (const std::string_view type = socketTypeName(socket->type); components == 1 && type != "VALUE")
        values = toScalar(type, values);
    return values;
}

std::optional<Core::float3> Folder::average(NodeTextureSource& source, const SocketRef socket, const std::size_t components) {
    textures = &source;
    const Samples values = evaluate(socket, components);
    if (unknown)
        return std::nullopt;
    Core::float3 sum;
    for (const Core::float4 value : values)
        sum = sum + Core::float3{value.x, value.y, value.z};
    const Core::float3 mean = sum / static_cast<float>(count);
    return components == 1 ? Core::float3{mean.x, 0.0f, 0.0f} : mean;
}

std::optional<Folder::Ramp> Folder::ramp(NodeTextureSource& source, const SocketRef socket, const std::size_t components, const GridMaximum* grids) {
    textures = &source;
    this->grids = grids;
    const Samples values = evaluate(socket, components);
    this->grids = nullptr;
    if (unknown)
        return std::nullopt;
    Ramp result{.grid = gridName, .values = {}};
    if (gridName.empty())
        return result;
    result.values.reserve(count);
    for (const Core::float4 value : values)
        result.values.push_back(components == 1 ? Core::float3{value.x, value.x, value.x} : Core::float3{value.x, value.y, value.z});
    return result;
}

Folder::Samples Folder::grid(const std::string& name) {
    const std::optional<float> maximum = (*grids)(name);
    if (!maximum)
        return constant({0.0f, 0.0f, 0.0f, 1.0f});
    if (!gridName.empty() && gridName != name)
        unknown = true;
    gridName = name;
    Samples values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float value = *maximum * float(index) / float(count - 1);
        values[index] = {value, value, value, 1.0f};
    }
    return values;
}

Folder::Samples Folder::socketDefault(const SocketRef socket) {
    if (!socket)
        return constant({});
    const NodeValues value = socketValues(socket);
    if (value.count == 1)
        return constant({value.values[0], value.values[0], value.values[0], 1.0f});
    Core::float4 padded{value.values[0], value.values[1], value.values[2], value.values[3]};
    if (value.count == 3 && socket->type == NodeSocketType::Color)
        padded.w = 1.0f;
    return constant(padded);
}

Folder::Samples Folder::input(const SocketRef socket) {
    const NodeLinkEnd link = links.linkedNode(socket);
    if (!link.node)
        return socketDefault(socket);
    Samples values = output(link.node, link.socket);
    const std::string_view to = socketTypeName(socket->type);
    const std::string_view from = socketTypeName(link.socket->type);
    if ((to == "VALUE" || to == "INT" || to == "BOOLEAN") && (from == "RGBA" || from == "VECTOR"))
        return toScalar(from, values);
    if (to == "RGBA" && (from == "VALUE" || from == "INT"))
        return splat(component(values, 0));
    return values;
}

std::vector<float> Folder::scalar(const NodeRef node, const char* name) {
    return component(input(inputNamed(node, name)), 0);
}

std::vector<float> Folder::noise(const float spread) {
    std::normal_distribution<float> distribution(0.5f, spread);
    std::vector<float> result(count, 0.5f);
    if (grids)
        return result;
    for (float& value : result)
        value = std::clamp(distribution(random), 0.0f, 1.0f);
    return result;
}

Folder::Samples Folder::mix(const NodeRef node) {
    const std::string type = text(node, "data_type");
    const std::string suffix = type == "RGBA" ? "_Color" : type == "VECTOR" ? "_Vector" : "_Float";
    const auto sockets = [&](const std::string& identifier) { return Materials::input(node, identifier); };
    const bool uniform = type != "VECTOR" || text(node, "factor_mode") == "UNIFORM";
    std::vector<float> factor = component(input(sockets(uniform ? "Factor_Float" : "Factor_Vector")), 0);
    if (flag(node, "clamp_factor"))
        for (float& value : factor)
            value = std::clamp(value, 0.0f, 1.0f);
    const Samples a = input(sockets("A" + suffix));
    const Samples b = input(sockets("B" + suffix));
    Samples result(count);
    const std::string mode = text(node, "blend_type");
    const bool clampResult = type == "RGBA" && flag(node, "clamp_result");
    for (std::size_t index = 0; index < count; ++index) {
        if (type != "RGBA") {
            result[index] = lerp(a[index], b[index], factor[index]);
            continue;
        }
        result[index] = blend(mode, factor[index], a[index], b[index]);
        result[index].w = a[index].w;
        if (clampResult)
            result[index] = clamp01(result[index]);
    }
    return result;
}

Folder::Samples Folder::output(const NodeRef node, const SocketRef socket) {
    if (unknown)
        return constant({});
    const std::string_view kind = nodeType(node);
    const std::string name = socket->name;
    if (kind == "RGB" || kind == "VALUE")
        return socketDefault(socket);
    if (kind == "TEX_IMAGE") {
        const std::span<const Core::float4> texels = text(node, "image").empty() ? std::span<const Core::float4>() : textures->samples(*node.tree, node.index);
        if (texels.size() != count) {
            unknown = true;
            return constant({});
        }
        const Samples values(texels.begin(), texels.end());
        return name == "Alpha" ? splat(component(values, 3)) : values;
    }
    if (isTexture(kind)) {
        const float spread = kind == "TEX_GRADIENT" || kind == "TEX_WHITE_NOISE" ? 0.29f : 0.12f;
        Samples values = constant({1.0f, 1.0f, 1.0f, 1.0f});
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const std::vector<float> random = noise(spread);
            for (std::size_t index = 0; index < count; ++index)
                (&values[index].x)[channel] = random[index];
        }
        if (socket->type != NodeSocketType::Color)
            for (Core::float4& value : values)
                value.y = value.z = value.x;
        return values;
    }
    if (grids && (kind == "ATTRIBUTE" || kind == "VOLUME_INFO")) {
        if (name == "Alpha")
            return constant({1.0f, 1.0f, 1.0f, 1.0f});
        constexpr std::pair<std::string_view, std::string_view> outputs[] = {
            {"Density", "density"}, {"Flame", "flame"}, {"Temperature", "temperature"}, {"Color", "color"}};
        if (kind == "ATTRIBUTE")
            return grid(text(node, "attribute_name"));
        for (const auto& [output, attribute] : outputs)
            if (name == output)
                return grid(std::string(attribute));
    }
    if (isCoordinate(kind))
        return constant({0.5f, 0.5f, 0.5f, 1.0f});
    if (kind == "VALTORGB") {
        const SocketRef fac = inputNamed(node, "Fac");
        const std::vector<float> factor = component(input(fac ? fac : inputAt(node, 0)), 0);
        const std::span<const float> stored = numbers(node, "color_ramp.elements");
        std::vector<std::pair<float, Core::float4>> elements;
        for (std::size_t index = 0; index + rampElementSize <= stored.size(); index += rampElementSize)
            elements.emplace_back(stored[index], Core::float4{stored[index + 1], stored[index + 2], stored[index + 3], stored[index + 4]});
        if (elements.empty()) {
            unknown = true;
            return constant({});
        }
        std::ranges::stable_sort(elements, {}, &std::pair<float, Core::float4>::first);
        const std::string interpolation = text(node, "color_ramp.interpolation");
        Samples colors(count);
        for (std::size_t sample = 0; sample < count; ++sample) {
            const float t = std::clamp(factor[sample], 0.0f, 1.0f);
            const auto above = std::ranges::upper_bound(elements, t, {}, &std::pair<float, Core::float4>::first);
            const std::size_t index = std::clamp<std::size_t>(static_cast<std::size_t>(above - elements.begin()), 1, elements.size()) - 1;
            if (interpolation == "CONSTANT" || elements.size() == 1) {
                colors[sample] = elements[index].second;
                continue;
            }
            const std::size_t upper = std::min(index + 1, elements.size() - 1);
            const float span = std::max(elements[upper].first - elements[index].first, 1e-8f);
            float weight = std::clamp((t - elements[index].first) / span, 0.0f, 1.0f);
            if (t < elements.front().first)
                weight = 0.0f;
            if (interpolation == "EASE")
                weight = weight * weight * (3.0f - 2.0f * weight);
            colors[sample] = lerp(elements[index].second, elements[upper].second, weight);
        }
        return name == "Alpha" ? splat(component(colors, 3)) : colors;
    }
    if (kind == "RGBTOBW")
        return splat(luminances(input(inputAt(node, 0))));
    if (kind == "MIX")
        return mix(node);
    if (kind == "MIX_RGB") {
        const std::vector<float> factor = component(input(inputAt(node, 0)), 0);
        const Samples a = input(inputAt(node, 1));
        const Samples b = input(inputAt(node, 2));
        const std::string mode = text(node, "blend_type");
        const bool clamped = flag(node, "use_clamp");
        Samples result(count);
        for (std::size_t index = 0; index < count; ++index) {
            result[index] = blend(mode, factor[index], a[index], b[index]);
            if (clamped)
                result[index] = clamp01(result[index]);
        }
        return result;
    }
    if (kind == "MATH") {
        const auto operation = mathOperation(text(node, "operation"));
        if (!operation) {
            unknown = true;
            return constant({});
        }
        std::vector<std::vector<float>> operands;
        for (const SocketRef operand : inputs(node))
            operands.push_back(component(input(operand), 0));
        operands.resize(3, std::vector<float>(count, 0.0f));
        const bool clamped = flag(node, "use_clamp");
        std::vector<float> result(count);
        for (std::size_t index = 0; index < count; ++index) {
            result[index] = (*operation)(operands[0][index], operands[1][index], operands[2][index]);
            if (clamped)
                result[index] = std::clamp(result[index], 0.0f, 1.0f);
        }
        return splat(result);
    }
    if (kind == "HUE_SAT") {
        const std::vector<float> hue = scalar(node, "Hue");
        const std::vector<float> saturation = scalar(node, "Saturation");
        const std::vector<float> value = scalar(node, "Value");
        const std::vector<float> factor = scalar(node, "Fac");
        Samples colors = input(inputNamed(node, "Color"));
        for (std::size_t index = 0; index < count; ++index) {
            const Core::float4 color = colors[index];
            Core::float3 hsv = rgbToHsv({std::max(color.x, 0.0f), std::max(color.y, 0.0f), std::max(color.z, 0.0f)});
            const float shifted = hsv.x + hue[index] - 0.5f;
            hsv = {shifted - std::floor(shifted), std::clamp(hsv.y * saturation[index], 0.0f, 1.0f), hsv.z * value[index]};
            const Core::float3 rgb = hsvToRgb(hsv);
            colors[index] = lerp(color, {rgb.x, rgb.y, rgb.z, color.w}, factor[index]);
        }
        return colors;
    }
    if (kind == "BRIGHTCONTRAST") {
        Samples colors = input(inputNamed(node, "Color"));
        const std::vector<float> bright = scalar(node, "Bright");
        const std::vector<float> contrast = scalar(node, "Contrast");
        for (std::size_t index = 0; index < count; ++index) {
            const float gain = 1.0f + contrast[index];
            const float offset = bright[index] - contrast[index] * 0.5f;
            Core::float4& color = colors[index];
            color = {std::max(gain * color.x + offset, 0.0f), std::max(gain * color.y + offset, 0.0f), std::max(gain * color.z + offset, 0.0f), color.w};
        }
        return colors;
    }
    if (kind == "INVERT") {
        Samples colors = input(inputNamed(node, "Color"));
        const std::vector<float> factor = scalar(node, "Fac");
        for (std::size_t index = 0; index < count; ++index) {
            const Core::float4 color = colors[index];
            colors[index] = lerp(color, {1.0f - color.x, 1.0f - color.y, 1.0f - color.z, color.w}, factor[index]);
        }
        return colors;
    }
    if (kind == "GAMMA") {
        Samples colors = input(inputNamed(node, "Color"));
        const std::vector<float> gamma = scalar(node, "Gamma");
        for (std::size_t index = 0; index < count; ++index) {
            Core::float4& color = colors[index];
            const float exponent = gamma[index];
            color = {std::pow(std::max(color.x, 0.0f), exponent), std::pow(std::max(color.y, 0.0f), exponent), std::pow(std::max(color.z, 0.0f), exponent),
                     color.w};
        }
        return colors;
    }
    if (kind == "CLAMP") {
        std::vector<float> value = scalar(node, "Value");
        const std::vector<float> low = scalar(node, "Min");
        const std::vector<float> high = scalar(node, "Max");
        for (std::size_t index = 0; index < count; ++index)
            value[index] = std::min(std::max(value[index], low[index]), high[index]);
        return splat(value);
    }
    if (kind == "MAP_RANGE" && text(node, "data_type") == "FLOAT") {
        std::vector<float> value = scalar(node, "Value");
        const std::vector<float> fromLow = scalar(node, "From Min");
        const std::vector<float> fromHigh = scalar(node, "From Max");
        const std::vector<float> toLow = scalar(node, "To Min");
        const std::vector<float> toHigh = scalar(node, "To Max");
        const bool clamped = flag(node, "clamp");
        for (std::size_t index = 0; index < count; ++index) {
            const float range = fromHigh[index] != fromLow[index] ? fromHigh[index] - fromLow[index] : 1.0f;
            float t = (value[index] - fromLow[index]) / range;
            if (clamped)
                t = std::clamp(t, 0.0f, 1.0f);
            value[index] = toLow[index] + t * (toHigh[index] - toLow[index]);
        }
        return splat(value);
    }
    if (kind == "SEPARATE_COLOR" || kind == "SEPRGB" || kind == "SEPXYZ") {
        std::size_t channel = 0;
        for (const SocketRef candidate : outputs(node)) {
            if (!candidate->enabled)
                continue;
            if (candidate == socket)
                break;
            ++channel;
        }
        return splat(component(input(inputAt(node, 0)), std::min<std::size_t>(channel, 3)));
    }
    if (kind == "COMBINE_COLOR" || kind == "COMBRGB" || kind == "COMBXYZ") {
        Samples values = constant({1.0f, 1.0f, 1.0f, 1.0f});
        std::size_t channel = 0;
        for (const SocketRef candidate : inputs(node)) {
            if (channel == 3)
                break;
            if (!candidate->enabled)
                continue;
            const std::vector<float> channelValues = component(input(candidate), 0);
            for (std::size_t index = 0; index < count; ++index)
                (&values[index].x)[channel] = channelValues[index];
            ++channel;
        }
        return values;
    }
    unknown = true;
    return constant({});
}

}
