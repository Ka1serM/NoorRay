#pragma once
#include "Raytracer.h"

class CpuRaytracer : public Raytracer  {

    
public:
    CpuRaytracer(Scene& scene, const uint32_t width, const uint32_t height)
        : Raytracer(scene, width, height)
    {
    }
};

