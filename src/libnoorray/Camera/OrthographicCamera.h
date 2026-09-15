#pragma once

#include "Camera/Camera.h"

class OrthographicCamera : public Camera {
public:
    OrthographicCamera();
    explicit OrthographicCamera(std::unique_ptr<Sensor> sensor);
    OrthographicCamera(const OrthographicCamera& other);
    ~OrthographicCamera();
    bool renderUi();
};
