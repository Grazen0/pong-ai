#include "frontend/state.h"
#include <SDL3_ttf/SDL_ttf.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include "frontend/util.h"

namespace frontend::state {

    GameState::GameState(TTF_Font* const display_font)
        : display_font(display_font) {}

    auto GameState::random() -> double {
        return dist(mt);
    }

    auto GameState::get_quit() const -> bool {
        return quit;
    }

    void GameState::reset_ball() {
        ball_x = (WINDOW_WIDTH / 2.0) - (BALL_SIZE / 2.0);
        ball_y = (WINDOW_HEIGHT / 2.0) - (BALL_SIZE / 2.0);
        ball_speed_x = 50;
        ball_speed_y = random() * 50;

        if (random() > 0.5) {
            ball_speed_y *= -1;
        }
    }

    void GameState::handle_event(const SDL_Event& event) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_KEY_DOWN:
                if ((event.key.mod & SDL_KMOD_CTRL) != 0 && event.key.key == SDLK_R) {
                    std::cout << R"(
 ____  _____     ___    ____   ____  ___  ____
|  _ \|_ _\ \   / / \  / ___| / ___|/ _ \|  _ \
| |_) || | \ \ / / _ \ \___ \| |  _| | | | | | |
|  _ < | |  \ V / ___ \ ___) | |_| | |_| | |_| |
|_| \_\___|  \_/_/   \_\____/ \____|\___/|____/
)";
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
            paddle_b_y -= PADDLE_SPEED * delta;
        }

        if (keyboard.contains(SDLK_DOWN) || keyboard.contains(SDLK_J) ||
            keyboard.contains(SDLK_S)) {
            paddle_a_y += PADDLE_SPEED * delta;
            paddle_b_y += PADDLE_SPEED * delta;
        }

        paddle_a_y =
            std::clamp(paddle_a_y, 0.0, static_cast<double>(WINDOW_HEIGHT - PADDLE_HEIGHT));
        paddle_b_y =
            std::clamp(paddle_b_y, 0.0, static_cast<double>(WINDOW_HEIGHT - PADDLE_HEIGHT));

        ball_x += ball_speed_x * delta;
        ball_y += ball_speed_y * delta;

        if (ball_y > WINDOW_HEIGHT - BALL_SIZE) {
            ball_y = WINDOW_HEIGHT - BALL_SIZE;
            ball_speed_y *= -1;
        } else

            if (ball_y <= 0) {
            ball_y = 0;
            ball_speed_y *= -1;
        }

        if ((ball_speed_x > 0 &&
             ball_x >= WINDOW_WIDTH - HORIZONTAL_PADDING - PADDLE_WIDTH - BALL_SIZE &&
             ball_y > paddle_a_y - BALL_SIZE && ball_y <= paddle_a_y + PADDLE_HEIGHT) ||
            (ball_speed_x < 0 && ball_x <= HORIZONTAL_PADDING + PADDLE_WIDTH &&
             ball_y > paddle_b_y - BALL_SIZE && ball_y <= paddle_b_y + PADDLE_HEIGHT)) {
            ball_speed_x *= -1;
        }

        if (ball_x >= WINDOW_WIDTH - BALL_SIZE) {
            ++score_b;
            reset_ball();
        } else if (ball_x <= 0) {
            ++score_a;
            reset_ball();
        }
    }

    void GameState::render(SDL_Renderer* const renderer) const {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        constexpr int DIVIDERS = 10;
        constexpr float DIVIDER_WIDTH = 10.0;
        constexpr float DIVIDER_HEIGHT = static_cast<float>(WINDOW_HEIGHT) / (2.0 * DIVIDERS);

        SDL_FRect divider_rect;
        divider_rect.x = (WINDOW_WIDTH / 2.0) - (DIVIDER_WIDTH / 2.0);
        divider_rect.w = DIVIDER_WIDTH;
        divider_rect.h = DIVIDER_HEIGHT;

        for (int i = 0; i < DIVIDERS; ++i) {
            divider_rect.y = (static_cast<float>(i) / DIVIDERS) * WINDOW_HEIGHT;
            SDL_RenderFillRect(renderer, &divider_rect);
        }

        SDL_FRect ball_rect;
        ball_rect.x = static_cast<float>(ball_x);
        ball_rect.y = static_cast<float>(ball_y);
        ball_rect.w = BALL_SIZE;
        ball_rect.h = BALL_SIZE;
        SDL_RenderFillRect(renderer, &ball_rect);

        SDL_FRect paddle_a_rect;
        paddle_a_rect.x = static_cast<float>(WINDOW_WIDTH - PADDLE_WIDTH - HORIZONTAL_PADDING);
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
        score_a_rect.x = (WINDOW_WIDTH / 2.0) + 30.0;
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
        score_b_rect.x = (WINDOW_WIDTH / 2.0F) - 30.0F - score_b_rect.w;
        score_b_rect.y = 20.0F;
        SDL_RenderTexture(renderer, score_b_texture, nullptr, &score_b_rect);

        SDL_DestroySurface(score_a_surface);
        SDL_DestroyTexture(score_a_texture);
        SDL_DestroySurface(score_b_surface);
        SDL_DestroyTexture(score_b_texture);

        SDL_RenderPresent(renderer);
    }

}  // namespace frontend::state
