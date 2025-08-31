#include "rendering.h"
#include "lidar_system.h"
#include <stdlib.h>
#include <time.h>
#include "lidar_denoiser_integration.h"

#define RAY_COUNT 60
#define NOISE_LEVEL 10.0

int main(int argc, char* argv[]) {
    srand(time(NULL));
    initSDL();

    LidarSystem* lidarSystem = createLidarSystem(100, RAY_COUNT, true);
    setNoiseLevel(lidarSystem, NOISE_LEVEL);

    // Add shapes to the environment
    addSquareToSystem(lidarSystem, 200, 150, 80, (SDL_Color){255, 100, 100, 255});
    addCircleToSystem(lidarSystem, 400, 200, 60, (SDL_Color){100, 255, 100, 255});
    addLineToSystem(lidarSystem, 150, 400, 550, 400, (SDL_Color){100, 100, 255, 255});
    addLineToSystem(lidarSystem, 600, 100, 800, 300, (SDL_Color){100, 100, 255, 255});
    addSquareToSystem(lidarSystem, 700, 500, 100, (SDL_Color){255, 255, 100, 255});
    addCircleToSystem(lidarSystem, 300, 300, 50, (SDL_Color){100, 255, 255, 255});

    // Add boundary walls
    addLineToSystem(lidarSystem, 0, 0, SPACE_WIDTH, 0, (SDL_Color){255, 255, 255, 255});
    addLineToSystem(lidarSystem, SPACE_WIDTH, 0, SPACE_WIDTH, SPACE_HEIGHT, (SDL_Color){255, 255, 255, 255});
    addLineToSystem(lidarSystem, SPACE_WIDTH, SPACE_HEIGHT, 0, SPACE_HEIGHT, (SDL_Color){255, 255, 255, 255});
    addLineToSystem(lidarSystem, 0, SPACE_HEIGHT, 0, 0, (SDL_Color){255, 255, 255, 255});


    // Initialize denoiser
    if (!init_lidar_denoiser("models/lidar_denoiser_traced.pt")) {
        // Handle error
        return 1;
    }

    while (running) {
        handleInput(lidarSystem);
        updateRays(lidarSystem);
        denoiseRays(lidarSystem);

        if (lidarSystem->render) {
            renderSpace(lidarSystem);
            renderGraph(lidarSystem);
            SDL_RenderPresent(renderer);
        }
        
        SDL_Delay(16);
    }

    destroyLidarSystem(lidarSystem);
    cleanup();
    return 0;
}