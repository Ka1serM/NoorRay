#pragma once
#include <iostream>

#if !defined(NDEBUG)   // Debug build
    #define LOG_INFO(x)   std::cout << "[INFO] "  << x << std::endl
    #define LOG_WARN(x)   std::cout << "[WARN] "  << x << std::endl
    #define LOG_ERROR(x)  std::cerr << "[ERROR] " << x << std::endl
    #define LOG_DEBUG(x)  std::cout << "[DEBUG] " << x << std::endl
#else                   // Release build
    #define LOG_INFO(x)   ((void)0)
    #define LOG_WARN(x)   ((void)0)
    #define LOG_ERROR(x)  std::cerr << "[ERROR] " << x << std::endl
    #define LOG_DEBUG(x)  ((void)0)
#endif
#define LOG_FATAL(x) std::cerr << "[FATAL] " << x << std::endl
