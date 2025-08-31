#ifndef LIDAR_SYSTEM_H
#define LIDAR_SYSTEM_H

#include <SDL.h>
#include <stdbool.h>
#include <float.h>
#include <time.h>

#define SPACE_WIDTH 1000
#define SPACE_HEIGHT 1200
#define PI 3.14159265358979323846

typedef enum {
    SHAPE_SQUARE,
    SHAPE_CIRCLE,
    SHAPE_LINE
} ShapeType;

typedef struct {
    ShapeType type;
    SDL_Rect rect;
    SDL_Color color;
    int radius;
    SDL_Point start;
    SDL_Point end;
} Shape;

typedef struct {
    SDL_Point origin;
    SDL_Point end_point;
    SDL_Point measured_point;
    SDL_Point predicted_point;
    
    double angle;
    double true_distance;    // Actual distance without noise
    double measured_distance; // Distance with noise applied
    double predicted_distance; // Distance predicted by the model
    SDL_Color color;
} Ray;

typedef struct {
    Shape* shapes;
    int shapeCount;
    int maxShapes;
    
    Ray* rays;
    int rayCount;
    
    SDL_Point lidarPosition;
    
    bool render;

    double noiseStdDev;
} LidarSystem;

// Lidar system functions
LidarSystem* createLidarSystem(int maxShapes, int rayCount, bool render);
void destroyLidarSystem(LidarSystem* system);
void addSquareToSystem(LidarSystem* system, int x, int y, int size, SDL_Color color);
void addCircleToSystem(LidarSystem* system, int x, int y, int radius, SDL_Color color);
void addLineToSystem(LidarSystem* system, int x1, int y1, int x2, int y2, SDL_Color color);
void setLidarPosition(LidarSystem* system, int x, int y);
void setNoiseLevel(LidarSystem* system, double stdDev);
void updateRays(LidarSystem* system);
bool rayIntersectsShape(Ray* ray, Shape* shape, double* distance, SDL_Point* intersection);
bool rayIntersectsCircle(Ray* ray, Shape* circle, double* distance, SDL_Point* intersection);
bool rayIntersectsSquare(Ray* ray, Shape* square, double* distance, SDL_Point* intersection);
bool rayIntersectsLine(Ray* ray, Shape* line, double* distance, SDL_Point* intersection);

// Helper function for generating Gaussian noise
double generateGaussianNoise(double mean, double stdDev);

#endif