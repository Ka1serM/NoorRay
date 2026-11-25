#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//! The version of this interface; the plugin refuses a bridge with another one.
#define NOORRAY_COLMAP_BRIDGE_VERSION 1
#define NOORRAY_COLMAP_EXPORT __attribute__((visibility("default")))

typedef enum NoorRayColmapResult { NoorRayColmapSucceeded = 0, NoorRayColmapFailed = 1, NoorRayColmapStopped = 2 } NoorRayColmapResult;

typedef struct NoorRayColmapCallbacks {
    void* user;
    //! Gets each COLMAP log line, one call at a time, from any thread of COLMAP.
    void (*line)(void* user, const char* text, size_t size);
    //! Returns non-zero when COLMAP should stop; the global mapper only checks it after it ends.
    int (*stopped)(void* user);
} NoorRayColmapCallbacks;

typedef int (*NoorRayColmapVersionFunction)(void);
typedef NoorRayColmapResult (*NoorRayColmapRunFunction)(int argumentCount, const char* const* arguments, const NoorRayColmapCallbacks* callbacks);

NOORRAY_COLMAP_EXPORT int noorray_colmap_version(void);
//! Runs a command of the colmap tool in this process: arguments[0] is the command (feature_extractor, exhaustive_matcher, sequential_matcher, global_mapper or image_undistorter) and the rest are its options.
NOORRAY_COLMAP_EXPORT NoorRayColmapResult noorray_colmap_run(int argumentCount, const char* const* arguments, const NoorRayColmapCallbacks* callbacks);

#ifdef __cplusplus
}
#endif
