#pragma once

#include <span>
#include <vector>

namespace NoorRay::Scene {

//! One spherical surface of a lens prescription, front to back, in metres. A zero radius is the aperture stop.
struct LensElement {
    float curvatureRadius = 0.0f;
    float thickness = 0.0f;
    float ior = 1.0f;
    float abbe = 0.0f;
    float apertureRadius = 0.0f;
};

struct Lens {
    std::vector<LensElement> elements;

    bool isEmpty() const { return elements.empty(); }
    float getTotalLength() const {
        float total = 0.0f;
        for (const LensElement& element : elements)
            total += element.thickness;
        return total;
    }
};

}
