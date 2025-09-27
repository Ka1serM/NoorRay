#include <windows.h>

class WindowEffects
{
public:
    static void EnableMica(SDL_Window* window)
    {
        if (!window)
            return;

        void* hwnd_ptr = nullptr;
        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        hwnd_ptr = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
        if (!hwnd_ptr)
            return;

        HWND hwnd = static_cast<HWND>(hwnd_ptr);

        // Load DwmSetWindowAttribute dynamically
        HMODULE hDwm = LoadLibraryA("dwmapi.dll");
        if (!hDwm)
            return;

        typedef HRESULT (WINAPI *PFNDwmSetWindowAttribute)(HWND, DWORD, LPCVOID, DWORD);
        PFNDwmSetWindowAttribute pDwmSetWindowAttribute =
            (PFNDwmSetWindowAttribute)GetProcAddress(hDwm, "DwmSetWindowAttribute");

        if (!pDwmSetWindowAttribute)
            return;

        const int DWMWA_SYSTEMBACKDROP_TYPE = 38;
        const int DWMSBT_MAINWINDOW = 2; // Mica

        pDwmSetWindowAttribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, &DWMSBT_MAINWINDOW, sizeof(DWMSBT_MAINWINDOW));
    }
    
    static void EnableAcrylic(SDL_Window* window)
    {
        if (!window) return;
        void* hwnd_ptr = nullptr;
        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        hwnd_ptr = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
        if (!hwnd_ptr) return;
        HWND hwnd = static_cast<HWND>(hwnd_ptr);

        // Load DwmSetWindowAttribute dynamically
        HMODULE hDwm = LoadLibraryA("dwmapi.dll");
        if (!hDwm)
            return;

        typedef HRESULT (WINAPI *PFNDwmSetWindowAttribute)(HWND, DWORD, LPCVOID, DWORD);
        PFNDwmSetWindowAttribute pDwmSetWindowAttribute = (PFNDwmSetWindowAttribute)GetProcAddress(hDwm, "DwmSetWindowAttribute");
        if (!pDwmSetWindowAttribute)
            return;

        const int DWMWA_SYSTEMBACKDROP_TYPE = 38;
        const int DWMSBT_TRANSIENTWINDOW = 3;
        // Acrylic
        pDwmSetWindowAttribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, &DWMSBT_TRANSIENTWINDOW, sizeof(DWMSBT_TRANSIENTWINDOW));
    };
};