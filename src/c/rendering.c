#include "rendering.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

SDL_Window* window = NULL;
SDL_Renderer* renderer = NULL;
bool running = true;
float dpi_scale_x = 1.0f;
float dpi_scale_y = 1.0f;

void initSDL() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        exit(1);
    }
    SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "permonitorv2");
    SDL_SetHint(SDL_HINT_WINDOWS_DPI_SCALING, "1");

    window = SDL_CreateWindow("Lidar System Simulation",
                             SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED,
                             SCREEN_WIDTH, SCREEN_HEIGHT,
                             SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI);
    
    if (window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        exit(1);
    }

    renderer = SDL_CreateRenderer(window, -1, 
                                 SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    if (renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        exit(1);
    }

    int renderW, renderH;
    SDL_GetRendererOutputSize(renderer, &renderW, &renderH);
    dpi_scale_x = (float)renderW / SCREEN_WIDTH;
    dpi_scale_y = (float)renderH / SCREEN_HEIGHT;

    printf("DPI Scale: %.2f x %.2f\n", dpi_scale_x, dpi_scale_y);
    
    SDL_RenderSetScale(renderer, 4.0f/dpi_scale_x, 4.0f/dpi_scale_y);
}

void cleanup() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}


void drawCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius, SDL_Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    
    if (radius == 0) {
        SDL_RenderDrawPoint(renderer, centerX, centerY);
    } else {
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                if (dx*dx + dy*dy <= radius*radius) {
                    SDL_RenderDrawPoint(renderer, centerX + dx, centerY + dy);
                }
            }
        }
    }
}

void renderSpace(LidarSystem* system) {
    SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
    SDL_RenderClear(renderer);
    
    SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
    SDL_Rect spaceRect = {0, 0, SPACE_WIDTH, SPACE_HEIGHT};
    SDL_RenderFillRect(renderer, &spaceRect);
    
    for (int i = 0; i < system->shapeCount; i++) {
        Shape shape = system->shapes[i];
        SDL_SetRenderDrawColor(renderer, shape.color.r, shape.color.g, shape.color.b, shape.color.a);
        
        switch (shape.type) {
            case SHAPE_SQUARE:
                SDL_RenderFillRect(renderer, &shape.rect);
                break;
            case SHAPE_CIRCLE:
                for (int y = -shape.radius; y <= shape.radius; y++) {
                    for (int x = -shape.radius; x <= shape.radius; x++) {
                        if (x*x + y*y <= shape.radius*shape.radius) {
                            SDL_RenderDrawPoint(renderer, shape.rect.x + x, shape.rect.y + y);
                        }
                    }
                }
                break;
            case SHAPE_LINE:
                SDL_RenderDrawLine(renderer, shape.start.x, shape.start.y, shape.end.x, shape.end.y);
                break;
        }
    }
    
    for (int i = 0; i < system->rayCount; i++) {
        Ray ray = system->rays[i];
        SDL_SetRenderDrawColor(renderer, ray.color.r, ray.color.g, ray.color.b, ray.color.a);
        SDL_RenderDrawLine(renderer, ray.origin.x, ray.origin.y, ray.end_point.x, ray.end_point.y);
        // Draw a blue circle at the measured point
        // drawCircle(renderer, ray.measured_point.x, ray.measured_point.y, 3, (SDL_Color){0, 0, 255, 255});
        // Draw a red circle at the true point
        // drawCircle(renderer, ray.end_point.x, ray.end_point.y, 3, (SDL_Color){255, 0, 0, 255});
        // Draw a green circle at the predicted point
        drawCircle(renderer, ray.predicted_point.x, ray.predicted_point.y, 3, (SDL_Color){0, 255, 0, 255});
    }
    
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Rect pointRect = {system->lidarPosition.x - 3, system->lidarPosition.y - 3, 6, 6};
    SDL_RenderFillRect(renderer, &pointRect);
}


void renderGraph(LidarSystem* system) {
    SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255);
    SDL_Rect graphRect = {SPACE_WIDTH, 0, GRAPH_WIDTH, SCREEN_HEIGHT};
    SDL_RenderFillRect(renderer, &graphRect);
    
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    SDL_RenderDrawLine(renderer, SPACE_WIDTH + 20, 20, SPACE_WIDTH + 20, SCREEN_HEIGHT - 20);
    SDL_RenderDrawLine(renderer, SPACE_WIDTH + 20, SCREEN_HEIGHT - 20, SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20);
    
    double maxDistance = fmax(SPACE_WIDTH, SPACE_HEIGHT);
    if (maxDistance < 1e-6) maxDistance = 1;
    
    int graphWidth = GRAPH_WIDTH - 40;
    int graphHeight = SCREEN_HEIGHT - 40;
    
    int radius = DISTANCE_GRAPH_POINT_DIAMETER / 2;
    
    // Define colors for different distance types
    SDL_Color trueColor = {255, 0, 0, 255};        // Red for true distance
    SDL_Color predictedColor = {0, 255, 0, 255};   // Green for predicted distance
    SDL_Color measuredColor = {0, 0, 255, 255};    // Blue for measured distance
    
    // First draw true distances (background)
    for (int i = 0; i < system->rayCount; i++) {
        double angle = system->rays[i].angle;
        double true_distance = system->rays[i].true_distance;
        
        int centerX = SPACE_WIDTH + 20 + (angle / (2 * PI)) * graphWidth;
        int centerY = SCREEN_HEIGHT - 20 - (true_distance / maxDistance) * graphHeight;
        
        drawCircle(renderer, centerX, centerY, radius, trueColor);
    }
    
    // Then draw predicted distances (middle layer)
    for (int i = 0; i < system->rayCount; i++) {
        double angle = system->rays[i].angle;
        double predicted_distance = system->rays[i].predicted_distance;
        
        int centerX = SPACE_WIDTH + 20 + (angle / (2 * PI)) * graphWidth;
        int centerY = SCREEN_HEIGHT - 20 - (predicted_distance / maxDistance) * graphHeight;
        
        drawCircle(renderer, centerX, centerY, radius, predictedColor);
    }
    
    // Finally draw measured distances (top layer)
    for (int i = 0; i < system->rayCount; i++) {
        double angle = system->rays[i].angle;
        double measured_distance = system->rays[i].measured_distance;
        
        int centerX = SPACE_WIDTH + 20 + (angle / (2 * PI)) * graphWidth;
        int centerY = SCREEN_HEIGHT - 20 - (measured_distance / maxDistance) * graphHeight;
        
        drawCircle(renderer, centerX, centerY, radius, measuredColor);
    }
}

void handleInput(LidarSystem* system) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_MOUSEMOTION:
                setLidarPosition(system, event.motion.x, event.motion.y);
                break;
        }
    }
}