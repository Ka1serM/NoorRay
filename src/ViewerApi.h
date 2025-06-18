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
#include "Shaders/SharedStructs.h"

#ifdef __cplusplus
extern "C" {
#endif

    VIEWERAPI_API void* init(int width, int height);

    VIEWERAPI_API void close();
  
    VIEWERAPI_API void add_mesh(Vertex* vertices, int vertexCount, uint32_t* indices, int indexCount);

#ifdef __cplusplus
}
#endif
