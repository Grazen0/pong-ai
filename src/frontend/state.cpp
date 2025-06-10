#include "frontend/state.h"
#include <SDL3_ttf/SDL_ttf.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include "frontend/constants.h"
#include "frontend/util.h"
#include "utec/algebra/vec2.h"

using utec::algebra::Vec2;

static constexpr utec::algebra::Vec2<int> BALL_SIZE{20, 20};
static constexpr int PADDLE_WIDTH = 20;
static constexpr int PADDLE_HEIGHT = 100;
static constexpr int PADDLE_SPEED = 100;
static constexpr int HORIZONTAL_PADDING = 20;

static constexpr float FONT_SCALING = 4.0F;
constexpr int DIVIDERS = 10;
constexpr float DIVIDER_WIDTH = 10.0;
constexpr float DIVIDER_HEIGHT = static_cast<float>(WINDOW_SIZE.y) / (2.0 * DIVIDERS);
constexpr Vec2<double> MAX_BALL_VELOCITY{200.0, 200.0};

namespace frontend::state {

    GameState::GameState(TTF_Font* const display_font)
        : display_font(display_font) {
        reset();
    }

    auto GameState::get_quit() const -> bool {
        return quit;
    }

    void GameState::reset() {
        paddle_a_y = (WINDOW_SIZE.y / 2.0) - (PADDLE_HEIGHT / 2.0);
        paddle_b_y = (WINDOW_SIZE.y / 2.0) - (PADDLE_HEIGHT / 2.0);
        score_a = 0;
        score_b = 0;
        reset_ball();
    }

    void GameState::reset_ball() {
        ball_position = static_cast<Vec2<double>>((WINDOW_SIZE / 2) - (BALL_SIZE / 2));
        ball_velocity.x = 50;
        ball_velocity.y = 50.0 * signed_unit_dist(mt);
    }

    void GameState::handle_event(const SDL_Event& event) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_R) {
                    if ((event.key.mod & SDL_KMOD_CTRL) != 0) {
                        std::cout << R"(
 ____  _____     ___    ____   ____  ___  ____
|  _ \|_ _\ \   / / \  / ___| / ___|/ _ \|  _ \
| |_) || | \ \ / / _ \ \___ \| |  _| | | | | | |
|  _ < | |  \ V / ___ \ ___) | |_| | |_| | |_| |
|_| \_\___|  \_/_/   \_\____/ \____|\___/|____/
)";
                    } else {
                        reset();
                    }
                }

                if (event.key.mod == 0) {
                    keyboard.insert(event.key.key);
                }
                break;
            case SDL_EVENT_KEY_UP:
                if (event.key.mod == 0) {
                    keyboard.erase(event.key.key);
                }
                break;
            default: {
            }
        }
    }

    void GameState::update(const double delta) {
        if (keyboard.contains(SDLK_UP) || keyboard.contains(SDLK_K) || keyboard.contains(SDLK_W)) {
            paddle_a_y -= PADDLE_SPEED * delta;
        }

        if (keyboard.contains(SDLK_DOWN) || keyboard.contains(SDLK_J) ||
            keyboard.contains(SDLK_S)) {
            paddle_a_y += PADDLE_SPEED * delta;
        }

        paddle_b_y = paddle_a_y;

        paddle_a_y =
            std::clamp(paddle_a_y, 0.0, static_cast<double>(WINDOW_SIZE.y - PADDLE_HEIGHT));
        paddle_b_y =
            std::clamp(paddle_b_y, 0.0, static_cast<double>(WINDOW_SIZE.y - PADDLE_HEIGHT));

        ball_position += ball_velocity * delta;

        if (ball_position.y > WINDOW_SIZE.y - BALL_SIZE.y) {
            ball_position.y = WINDOW_SIZE.y - BALL_SIZE.y;
            ball_velocity.y *= -1;
        } else if (ball_position.y <= 0) {
            ball_position.y = 0;
            ball_velocity.y *= -1;
        }

        if ((ball_velocity.x > 0 &&
             ball_position.x >= WINDOW_SIZE.x - HORIZONTAL_PADDING - PADDLE_WIDTH - BALL_SIZE.x &&
             ball_position.y > paddle_a_y - BALL_SIZE.y &&
             ball_position.y <= paddle_a_y + PADDLE_HEIGHT) ||
            (ball_velocity.x < 0 && ball_position.x <= HORIZONTAL_PADDING + PADDLE_WIDTH &&
             ball_position.y > paddle_b_y - BALL_SIZE.y &&
             ball_position.y <= paddle_b_y + PADDLE_HEIGHT)) {
            ball_velocity.x *= -1;
            ball_velocity.x -= 10.0 * static_cast<double>(std::signbit(ball_velocity.x));
            ball_velocity.y += 30.0 * signed_unit_dist(mt);

            if (std::abs(ball_velocity.y) > MAX_BALL_VELOCITY.y) {
                ball_velocity.y *= MAX_BALL_VELOCITY.y / ball_velocity.y;
            }

            if (std::abs(ball_velocity.x) > MAX_BALL_VELOCITY.x) {
                ball_velocity.x *= MAX_BALL_VELOCITY.x / ball_velocity.x;
            }
        }

        if (ball_position.x >= WINDOW_SIZE.x - BALL_SIZE.x) {
            ++score_b;
            reset_ball();
        } else if (ball_position.x <= 0) {
            ++score_a;
            reset_ball();
        }
    }

    void GameState::render(SDL_Renderer* const renderer) const {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        SDL_FRect divider_rect;
        divider_rect.x = (WINDOW_SIZE.x / 2.0) - (DIVIDER_WIDTH / 2.0);
        divider_rect.w = DIVIDER_WIDTH;
        divider_rect.h = DIVIDER_HEIGHT;

        for (int i = 0; i < DIVIDERS; ++i) {
            divider_rect.y = (static_cast<float>(i) / DIVIDERS) * WINDOW_SIZE.y;
            SDL_RenderFillRect(renderer, &divider_rect);
        }

        SDL_FRect ball_rect;
        ball_rect.x = static_cast<float>(ball_position.x);
        ball_rect.y = static_cast<float>(ball_position.y);
        ball_rect.w = BALL_SIZE.x;
        ball_rect.h = BALL_SIZE.y;
        SDL_RenderFillRect(renderer, &ball_rect);

        SDL_FRect paddle_a_rect;
        paddle_a_rect.x = static_cast<float>(WINDOW_SIZE.x - PADDLE_WIDTH - HORIZONTAL_PADDING);
        paddle_a_rect.y = static_cast<float>(paddle_a_y);
        paddle_a_rect.w = PADDLE_WIDTH;
        paddle_a_rect.h = PADDLE_HEIGHT;
        SDL_RenderFillRect(renderer, &paddle_a_rect);

        SDL_FRect paddle_b_rect;
        paddle_b_rect.x = static_cast<float>(HORIZONTAL_PADDING);
        paddle_b_rect.y = static_cast<float>(paddle_b_y);
        paddle_b_rect.w = PADDLE_WIDTH;
        paddle_b_rect.h = PADDLE_HEIGHT;
        SDL_RenderFillRect(renderer, &paddle_b_rect);

        const std::string score_a_str = frontend::util::pad_left(std::to_string(score_a), 2, '0');

        SDL_Surface* const score_a_surface = TTF_RenderText_Solid(
            display_font, score_a_str.c_str(), 0, {255, 255, 255, SDL_ALPHA_OPAQUE});
        SDL_Texture* const score_a_texture =
            SDL_CreateTextureFromSurface(renderer, score_a_surface);
        SDL_SetTextureScaleMode(score_a_texture, SDL_SCALEMODE_NEAREST);

        SDL_FRect score_a_rect;
        score_a_rect.w = static_cast<float>(score_a_texture->w) * FONT_SCALING;
        score_a_rect.h = static_cast<float>(score_a_texture->h) * FONT_SCALING;
        score_a_rect.x = (WINDOW_SIZE.x / 2.0) + 30.0;
        score_a_rect.y = 20.0F;
        SDL_RenderTexture(renderer, score_a_texture, nullptr, &score_a_rect);

        const std::string score_b_str = frontend::util::pad_left(std::to_string(score_b), 2, '0');

        SDL_Surface* const score_b_surface = TTF_RenderText_Solid(
            display_font, score_b_str.c_str(), 0, {255, 255, 255, SDL_ALPHA_OPAQUE});
        SDL_Texture* const score_b_texture =
            SDL_CreateTextureFromSurface(renderer, score_b_surface);
        SDL_SetTextureScaleMode(score_b_texture, SDL_SCALEMODE_NEAREST);

        SDL_FRect score_b_rect;
        score_b_rect.w = static_cast<float>(score_b_texture->w) * FONT_SCALING;
        score_b_rect.h = static_cast<float>(score_b_texture->h) * FONT_SCALING;
        score_b_rect.x = (WINDOW_SIZE.x / 2.0F) - 30.0F - score_b_rect.w;
        score_b_rect.y = 20.0F;
        SDL_RenderTexture(renderer, score_b_texture, nullptr, &score_b_rect);

        SDL_DestroySurface(score_a_surface);
        SDL_DestroyTexture(score_a_texture);
        SDL_DestroySurface(score_b_surface);
        SDL_DestroyTexture(score_b_texture);

        SDL_RenderPresent(renderer);
    }

}  // namespace frontend::state
