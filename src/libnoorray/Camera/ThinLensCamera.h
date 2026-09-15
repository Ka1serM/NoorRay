#pragma once

#include "Camera/Camera.h"

class ThinLensCamera : public Camera {
public:
    ThinLensCamera();
    explicit ThinLensCamera(std::unique_ptr<Sensor> sensor);
    ThinLensCamera(const ThinLensCamera& other);
    ~ThinLensCamera();
    float apertureDiameterMm{};
    float bokehBias{1.f};

    bool renderUi();
};
