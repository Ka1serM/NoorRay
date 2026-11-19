#pragma once

#include <array>
#include <optional>

#include <RmlUi/Core/Element.h>

#include "NoorRay/Core/Math.h"

namespace NoorRay::App {

//! The camera drags of the viewport: the view gizmo's ball orbits, its tools dolly and pan.
enum class ViewDrag { Orbit, Pan, Look, Dolly };

//! The view gizmo inside a viewport, after ImViewGuizmo: .view-gizmo holds .gizmo-center and .gizmo-rotor with the six .gizmo-axis handles (+X, -X, +Y, -Y, +Z, -Z) and the .spoke-x, .spoke-y and .spoke-z spokes; .view-tools holds .tool-dolly and .tool-pan.
class ViewGizmo {
public:
    //! Turns the rotor to the camera; the stylesheet sizes everything, so only transforms and order classes change.
    void sync(Rml::Element& viewport, const Core::Transform& camera);
    void forget() { found = false; }

    //! The drag that a press on the element starts, if it is part of the gizmo.
    static std::optional<ViewDrag> dragOf(const Rml::Element* element);
    //! The axis handle the element belongs to, or -1.
    static int axisOf(const Rml::Element* element);

private:
    bool find(Rml::Element& viewport);

    bool found = false;
    Rml::Element* rotor = nullptr;
    std::array<Rml::Element*, 3> spokes{};
    std::array<Rml::Element*, 6> axes{};
    std::array<float, 9> basis{};
    std::array<int, 6> rank{};
    std::array<bool, 6> behind{};
    float ratio = 0.0f;
    bool valid = false;
};

}
