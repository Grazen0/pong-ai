#ifndef INCLUDE_FRONTEND_STATE_H
#define INCLUDE_FRONTEND_STATE_H

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_render.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <random>
#include <set>
#include "frontend/constants.h"

namespace frontend::state {
    class GameState {
        bool quit = false;
        std::set<SDL_Keycode> keyboard;
        TTF_Font* display_font;

        std::random_device rd;
        std::mt19937 mt{rd()};
        std::uniform_real_distribution<double> dist{0.0, 1.0};

        double ball_x = (WINDOW_WIDTH / 2.0) - (BALL_SIZE / 2.0);
        double ball_y = (WINDOW_HEIGHT / 2.0) - (BALL_SIZE / 2.0);
        double ball_speed_x = 50;
        double ball_speed_y = 20;

        double paddle_a_y = (WINDOW_HEIGHT / 2.0) - (PADDLE_HEIGHT / 2.0);
        double paddle_b_y = (WINDOW_HEIGHT / 2.0) - (PADDLE_HEIGHT / 2.0);
        int score_a = 0;
        int score_b = 0;

        [[nodiscard]] auto random() -> double;

    public:
        explicit GameState(TTF_Font* display_font);

        [[nodiscard]] auto get_quit() const -> bool;

        void reset_ball();

        void handle_event(const SDL_Event& event);

        void update(double delta);

        void render(SDL_Renderer* renderer) const;
    };

}  // namespace frontend::state

#endif
