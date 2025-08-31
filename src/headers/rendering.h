#ifndef RENDERING_H
#define RENDERING_H

#include <SDL.h>
#include <stdbool.h>
#include "lidar_system.h"

#define SCREEN_WIDTH 2000
#define SCREEN_HEIGHT 1200
#define GRAPH_WIDTH (SCREEN_WIDTH - SPACE_WIDTH)
#define DISTANCE_GRAPH_POINT_DIAMETER 4

extern float dpi_scale_x;
extern float dpi_scale_y;
extern SDL_Window* window;
extern SDL_Renderer* renderer;
extern bool running;

// Rendering functions
void initSDL();
void cleanup();
void renderSpace(LidarSystem* system);
void renderGraph(LidarSystem* system);
void handleInput(LidarSystem* system);

#endif