#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_oldnames.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "frontend/constants.h"
#include "frontend/font_data.h"
#include "frontend/state.h"
#include "frontend/util.h"

auto main() -> int {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Could not initialize video: " << SDL_GetError() << '\n';
        return 1;
    }

    if (!TTF_Init()) {
        std::cerr << "Could not init TTF: " << SDL_GetError() << '\n';
        return 1;
    }

    TTF_Font* const display_font = TTF_OpenFontIO(
        SDL_IOFromConstMem(static_cast<const void*>(DISPLAY_FONT_DATA), DISPLAY_FONT_DATA_LEN),
        true, 20.0F);

    if (display_font == nullptr) {
        std::cerr << "Could not load font: " << SDL_GetError() << '\n';
        return 1;
    }

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    if (!SDL_CreateWindowAndRenderer("Pong AI", WINDOW_SIZE.x, WINDOW_SIZE.y, 0, &window,
                                     &renderer)) {
        std::cerr << "Could not initialize window or renderer: " << SDL_GetError() << '\n';
        return 1;
    }

    SDL_RenderPresent(renderer);

    frontend::state::GameState state(display_font);

    long double last_time = frontend::util::get_performance_time();
    long double time_accumulator = 0.0;

    while (!state.get_quit()) {
        const long double frame_start = frontend::util::get_performance_time();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            state.handle_event(event);
        }

        const long double new_time = frontend::util::get_performance_time();
        time_accumulator = std::max(time_accumulator + new_time - last_time, MAX_TIME_ACCUMULATOR);
        last_time = new_time;

        while (time_accumulator >= DELTA) {
            state.update(DELTA);
            time_accumulator -= DELTA;
        }

        state.render(renderer);

        const long double offset = frontend::util::get_performance_time() - frame_start;
        const long double delay = DELTA - offset;
        if (delay > 0) {
            SDL_DelayNS(static_cast<std::uint64_t>(delay * 1e9));
        }
    }

    TTF_CloseFont(display_font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
