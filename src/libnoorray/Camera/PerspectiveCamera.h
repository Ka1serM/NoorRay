#pragma once

#include "Camera/Camera.h"

class PerspectiveCamera : public Camera {
public:
    PerspectiveCamera();
    explicit PerspectiveCamera(std::unique_ptr<Sensor> sensor);
    PerspectiveCamera(const PerspectiveCamera& other);
    ~PerspectiveCamera();
};
