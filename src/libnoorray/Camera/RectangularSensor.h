#pragma once

#include "Camera/Sensor.h"

class RectangularSensor : public Sensor {
public:
    RectangularSensor() = default;
    explicit RectangularSensor(const Sensor& other);
    ~RectangularSensor() override = default;
    bool renderUi(Sensor& owner);
};
