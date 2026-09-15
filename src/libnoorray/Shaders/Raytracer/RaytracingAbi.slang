#pragma once

#include "../../Shared/Raytracing.h"

[[vk::push_constant]] ConstantBuffer<RaytracerRoot> root;
