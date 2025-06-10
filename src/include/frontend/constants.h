#ifndef INCLUDE_FRONTEND_CONSTANTS_H
#define INCLUDE_FRONTEND_CONSTANTS_H

constexpr int WINDOW_WIDTH = 720;
constexpr int WINDOW_HEIGHT = 540;
constexpr int FPS = 60;
constexpr double DELTA = 1.0 / FPS;
constexpr long double MAX_TIME_ACCUMULATOR = 4 * DELTA;

constexpr float FONT_SCALING = 5.0F;

constexpr int BALL_SIZE = 20;
constexpr int PADDLE_WIDTH = 20;
constexpr int PADDLE_HEIGHT = 100;
constexpr int PADDLE_SPEED = 100;
constexpr int HORIZONTAL_PADDING = 20;

#endif
