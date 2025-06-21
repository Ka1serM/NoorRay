#pragma once

#ifdef _WIN32
  #ifdef VIEWERAPI_EXPORTS
    #define VIEWERAPI_API __declspec(dllexport)
  #else
    #define VIEWERAPI_API __declspec(dllimport)
  #endif
#else
  #define VIEWERAPI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    VIEWERAPI_API void init(int width, int height);
    VIEWERAPI_API void close();

#ifdef __cplusplus
}
#endif
