#ifndef INCLUDE_FRONTEND_STATE_H
#define INCLUDE_FRONTEND_STATE_H

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_render.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <random>
#include <set>
#include "utec/algebra/vec2.h"

namespace frontend::state {
    class GameState {
        bool quit = false;
        std::set<SDL_Keycode> keyboard;
        TTF_Font* display_font = nullptr;

        std::random_device rd;
        std::mt19937 mt{rd()};
        std::uniform_real_distribution<double> signed_unit_dist{-1.0, 1.0};

        utec::algebra::Vec2<double> ball_position{};
        utec::algebra::Vec2<double> ball_velocity{};

        double paddle_a_y{};
        double paddle_b_y{};
        int score_a{};
        int score_b{};

    public:
        explicit GameState(TTF_Font* display_font);

        [[nodiscard]] auto get_quit() const -> bool;

        void reset();

        void reset_ball();

        void handle_event(const SDL_Event& event);

        void update(double delta);

        void render(SDL_Renderer* renderer) const;
    };

}  // namespace frontend::state

#endif
