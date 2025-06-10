#ifndef INCLUDE_FRONTEND_UTIL_H
#define INCLUDE_FRONTEND_UTIL_H

#include <string>

namespace frontend::util {

    auto get_performance_time() -> long double;

    auto pad_left(std::string s, std::size_t max_len, char fill = ' ') -> std::string;

}  // namespace frontend::util

#endif
