#ifndef INCLUDE_FRONTEND_CONSTANTS_H
#define INCLUDE_FRONTEND_CONSTANTS_H

#include "utec/algebra/vec2.h"

constexpr utec::algebra::Vec2<int> WINDOW_SIZE{720, 540};
constexpr int FPS = 60;
constexpr double DELTA = 1.0 / FPS;
constexpr long double MAX_TIME_ACCUMULATOR = 4 * DELTA;

#endif
