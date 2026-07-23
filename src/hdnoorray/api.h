#pragma once

#include <pxr/base/arch/export.h>

#if defined(HDNOORRAY_EXPORTS)
#define HDNOORRAY_API ARCH_EXPORT
#else
#define HDNOORRAY_API ARCH_IMPORT
#endif
