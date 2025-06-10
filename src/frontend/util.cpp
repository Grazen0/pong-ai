#include "frontend/util.h"
#include <SDL3/SDL_timer.h>
#include <cstddef>
#include <string>

namespace frontend::util {

    auto get_performance_time() -> long double {
        return static_cast<long double>(SDL_GetPerformanceCounter()) /
               SDL_GetPerformanceFrequency();
    }

    auto pad_left(std::string s, const std::size_t max_len, const char fill) -> std::string {
        if (s.size() >= max_len) {
            return s;
        }

        return std::string(max_len - s.size(), fill) + std::move(s);
    }

}  // namespace frontend::util
