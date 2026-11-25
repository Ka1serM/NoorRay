#pragma once

#include "DiffRenderer/Core/Image.h"

namespace DiffRenderer::Core {

//! The mean SSIM over all pixels and channels, with an 11x11 Gaussian window of sigma 1.5 and zero padding, as the ssim of 3DGS.
double getSsim(const Image& image, const Image& reference);

}
