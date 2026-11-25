#include "NoorRay/Materials/IesProfile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <utility>

namespace NoorRay::Materials {

namespace {

constexpr float pi = std::numbers::pi_v<float>;
constexpr std::uint32_t tableWidth = 720;
constexpr std::uint32_t tableHeight = 360;
constexpr long maxAngles = 100000;

//! Cycles' IESTextParser: commas count as spaces, and the values start at the TILT line.
class Parser {
public:
    explicit Parser(const std::string_view source) : text(source) {
        std::ranges::replace(text, ',', ' ');
        const std::size_t tilt = text.find("\nTILT=");
        position = tilt == std::string::npos ? nullptr : text.data() + tilt;
    }
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    bool eof() const { return !position || *position == '\0'; }
    bool failed() const { return error; }

    double number() {
        if (eof())
            return fail();
        char* end = nullptr;
        const double value = std::strtod(position, &end);
        if (end == position)
            return fail();
        position = end;
        return value;
    }

    long integer() {
        if (eof())
            return static_cast<long>(fail());
        char* end = nullptr;
        const long value = std::strtol(position, &end, 10);
        if (end == position)
            return static_cast<long>(fail());
        position = end;
        return value;
    }

    std::string text;
    const char* position = nullptr;

private:
    double fail() {
        position = nullptr;
        error = true;
        return 0.0;
    }

    bool error = false;
};

using Rows = std::vector<std::vector<float>>;

bool angleClose(const float a, const float b) {
    return std::abs(a - b) < 1e-4f;
}

//! Cycles' process_type_a: the vertical angles move to start at the nadir, and the horizontal angles mirror around 180 degrees.
void processTypeA(std::vector<float>& horizontal, std::vector<float>& vertical, Rows& intensity) {
    for (float& angle : vertical)
        angle += 90.0f;
    std::vector<float> angles;
    Rows rows;
    for (std::size_t i = horizontal.size(); i-- > 0;) {
        angles.push_back(180.0f - horizontal[i]);
        rows.push_back(intensity[i]);
    }
    if (angleClose(horizontal[0], 0.0f))
        for (std::size_t i = 1; i < horizontal.size(); ++i) {
            angles.push_back(180.0f + horizontal[i]);
            rows.push_back(intensity[i]);
        }
    horizontal = std::move(angles);
    intensity = std::move(rows);
}

//! Cycles' process_type_b: the two angles swap, and each half range mirrors to a full range.
void processTypeB(std::vector<float>& horizontal, std::vector<float>& vertical, Rows& intensity) {
    Rows swapped(vertical.size());
    for (std::size_t i = 0; i < vertical.size(); ++i)
        for (std::size_t j = 0; j < horizontal.size(); ++j)
            swapped[i].push_back(intensity[j][i]);
    intensity = std::move(swapped);
    std::swap(horizontal, vertical);
    if (angleClose(horizontal[0], 0.0f)) {
        std::vector<float> angles;
        Rows rows;
        for (std::size_t i = horizontal.size() - 1; i > 0; --i) {
            angles.push_back(90.0f - horizontal[i]);
            rows.push_back(intensity[i]);
        }
        for (std::size_t i = 0; i < horizontal.size(); ++i) {
            angles.push_back(90.0f + horizontal[i]);
            rows.push_back(intensity[i]);
        }
        horizontal = std::move(angles);
        intensity = std::move(rows);
    } else {
        for (float& angle : horizontal)
            angle += 90.0f;
    }
    if (!angleClose(vertical[0], 0.0f)) {
        for (float& angle : vertical)
            angle += 90.0f;
        return;
    }
    std::vector<float> angles;
    for (std::size_t i = vertical.size() - 1; i > 0; --i)
        angles.push_back(90.0f - vertical[i]);
    for (const float angle : vertical)
        angles.push_back(90.0f + angle);
    for (std::vector<float>& row : intensity) {
        std::vector<float> mirrored;
        for (std::size_t j = row.size() - 1; j > 0; --j)
            mirrored.push_back(row[j]);
        mirrored.insert(mirrored.end(), row.begin(), row.end());
        row = std::move(mirrored);
    }
    vertical = std::move(angles);
}

//! Cycles' process_type_c: the horizontal angles start at 0 and fill the full circle by symmetry.
void processTypeC(std::vector<float>& horizontal, Rows& intensity) {
    if (angleClose(horizontal[0], 90.0f))
        for (float& angle : horizontal)
            angle -= 90.0f;
    if (horizontal.size() == 1) {
        horizontal[0] = 0.0f;
        horizontal.push_back(360.0f);
        intensity.push_back(intensity[0]);
    }
    if (angleClose(horizontal.back(), 90.0f))
        for (std::size_t count = horizontal.size(), i = count - 1; i-- > 0;) {
            horizontal.push_back(180.0f - horizontal[i]);
            intensity.push_back(intensity[i]);
        }
    if (angleClose(horizontal.back(), 180.0f))
        for (std::size_t count = horizontal.size(), i = count - 1; i-- > 0;) {
            horizontal.push_back(360.0f - horizontal[i]);
            intensity.push_back(intensity[i]);
        }
    if (!angleClose(horizontal[0], 0.0f) || angleClose(horizontal.back(), 360.0f))
        return;
    const std::size_t count = horizontal.size();
    const float lastStep = horizontal[count - 1] - horizontal[count - 2];
    const float firstStep = horizontal[1] - horizontal[0];
    const float gapStep = 360.0f - horizontal[count - 1];
    if (!angleClose(lastStep, gapStep) && !angleClose(firstStep, gapStep))
        return;
    horizontal.push_back(360.0f);
    intensity.push_back(intensity[0]);
}

float cubic(const float a, const float b, const float c, const float d, const float x) {
    return 0.5f * (((d + 3.0f * (b - c) - a) * x + (2.0f * a - 5.0f * b + 4.0f * c - d)) * x + (c - a)) * x + b;
}

}

std::expected<IesProfile, Core::Error> parseIes(const std::string_view text) {
    const auto invalid = [](const std::string_view what) {
        return Core::fail(Core::ErrorCode::InvalidFile, std::string("the IES profile ") + std::string(what));
    };
    Parser parser(text);
    if (parser.eof())
        return invalid("has no TILT line");
    if (std::strncmp(parser.position, "\nTILT=INCLUDE", 13) == 0) {
        parser.position += 13;
        parser.number();
        const long tilts = parser.integer();
        for (long i = 0; i < 2 * tilts && !parser.failed(); ++i)
            parser.number();
    } else {
        parser.position = std::strchr(parser.position + 1, '\n');
    }
    if (parser.eof())
        return invalid("ends after the TILT line");
    ++parser.position;
    parser.integer();
    parser.number();
    double factor = parser.number();
    const long verticalCount = parser.integer();
    const long horizontalCount = parser.integer();
    const long type = parser.integer();
    if (type < 1 || type > 3)
        return invalid("has no photometric type A, B or C");
    if (verticalCount < 1 || horizontalCount < 1 || verticalCount > maxAngles || horizontalCount > maxAngles)
        return invalid("has no valid count of angles");
    parser.integer();
    parser.number();
    parser.number();
    parser.number();
    factor *= parser.number();
    factor *= parser.number();
    parser.number();
    //! Cycles' scale from candela to its light units.
    factor *= 0.0706650768394;
    IesProfile profile;
    for (long i = 0; i < verticalCount; ++i)
        profile.vertical.push_back(static_cast<float>(parser.number()));
    for (long i = 0; i < horizontalCount; ++i)
        profile.horizontal.push_back(static_cast<float>(parser.number()));
    Rows intensity(static_cast<std::size_t>(horizontalCount));
    for (std::vector<float>& row : intensity)
        for (long j = 0; j < verticalCount; ++j)
            row.push_back(static_cast<float>(factor * parser.number()));
    if (parser.failed())
        return invalid("ends before its values");
    if (type == 3)
        processTypeA(profile.horizontal, profile.vertical, intensity);
    else if (type == 2)
        processTypeB(profile.horizontal, profile.vertical, intensity);
    else
        processTypeC(profile.horizontal, intensity);
    for (float& angle : profile.vertical)
        angle *= pi / 180.0f;
    for (float& angle : profile.horizontal)
        angle *= pi / 180.0f;
    for (const std::vector<float>& row : intensity)
        profile.intensity.insert(profile.intensity.end(), row.begin(), row.end());
    return profile;
}

float IesProfile::evaluate(const float horizontalAngle, const float verticalAngle) const {
    const std::size_t horizontalCount = horizontal.size();
    const std::size_t verticalCount = vertical.size();
    if (verticalAngle < vertical.front() || verticalAngle >= vertical.back() || horizontalAngle < horizontal.front() || horizontalAngle >= horizontal.back())
        return 0.0f;
    const bool wrapsHorizontal = horizontal.front() < 1e-7f && horizontal.back() > 2.0f * pi - 1e-7f;
    const bool wrapsLow = vertical.front() < 1e-7f;
    const bool wrapsHigh = vertical.back() > pi - 1e-7f;
    std::size_t h = 0;
    while (horizontal[h + 1] < horizontalAngle)
        ++h;
    std::size_t v = 0;
    while (vertical[v + 1] < verticalAngle)
        ++v;
    const float horizontalFraction = (horizontalAngle - horizontal[h]) / (horizontal[h + 1] - horizontal[h]);
    const float verticalFraction = (verticalAngle - vertical[v]) / (vertical[v + 1] - vertical[v]);
    const auto at = [&](const std::size_t row, const std::size_t column) { return intensity[row * verticalCount + column]; };
    const auto column = [&](const std::size_t row) {
        const float c = at(row, v + 1);
        const float b = at(row, v);
        float a = b;
        if (v > 0)
            a = at(row, v - 1);
        else if (wrapsLow)
            a = at(row, 1);
        float d = c;
        if (v + 2 < verticalCount)
            d = at(row, v + 2);
        else if (wrapsHigh)
            d = at(row, verticalCount - 2);
        return cubic(a, b, c, d, verticalFraction);
    };
    const float b = column(h);
    const float c = column(h + 1);
    float a = b;
    if (h > 0)
        a = column(h - 1);
    else if (wrapsHorizontal)
        a = column(horizontalCount - 2);
    float d = b;
    if (h + 2 < horizontalCount)
        d = column(h + 2);
    else if (wrapsHorizontal)
        d = column(1);
    return std::max(cubic(a, b, c, d, horizontalFraction), 0.0f);
}

Scene::Texture makeIesTexture(const IesProfile& profile, std::string name) {
    Scene::Texture texture;
    texture.name = std::move(name);
    texture.format = NoorRHI::ImageFormat::R32Float;
    texture.width = tableWidth;
    texture.height = tableHeight;
    std::vector<float> values(std::size_t{tableWidth} * tableHeight);
    for (std::uint32_t row = 0; row < tableHeight; ++row) {
        const float verticalAngle = (1.0f - (static_cast<float>(row) + 0.5f) / tableHeight) * pi;
        for (std::uint32_t x = 0; x < tableWidth; ++x)
            values[std::size_t{row} * tableWidth + x] = profile.evaluate((static_cast<float>(x) + 0.5f) / tableWidth * 2.0f * pi, verticalAngle);
    }
    texture.data.resize(values.size() * sizeof(float));
    std::memcpy(texture.data.data(), values.data(), texture.data.size());
    texture.levels.push_back({.offset = 0, .size = texture.data.size(), .width = tableWidth, .height = tableHeight});
    return texture;
}

}
