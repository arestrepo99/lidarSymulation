#include "lidar_system.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

// Box-Muller transform for generating Gaussian noise
double generateGaussianNoise(double mean, double stdDev) {
    static int hasSpare = 0;
    static double spare;
    
    if (hasSpare) {
        hasSpare = 0;
        return mean + stdDev * spare;
    }
    
    hasSpare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stdDev * u * s;
}

LidarSystem* createLidarSystem(int maxShapes, int rayCount, bool render) {
    LidarSystem* system = (LidarSystem*)malloc(sizeof(LidarSystem));
    system->shapes = (Shape*)malloc(maxShapes * sizeof(Shape));
    system->shapeCount = 0;
    system->maxShapes = maxShapes;
    
    system->rays = (Ray*)malloc(rayCount * sizeof(Ray));
    system->rayCount = rayCount;
    
    system->lidarPosition.x = SPACE_WIDTH / 2;
    system->lidarPosition.y = SPACE_HEIGHT / 2;
    
    system->render = render;
    system->noiseStdDev = 0.0; // Default: no noise
    
    // Seed the random number generator
    srand(time(NULL));
    
    return system;
}

void destroyLidarSystem(LidarSystem* system) {
    free(system->shapes);
    free(system->rays);
    free(system);
}

void addSquareToSystem(LidarSystem* system, int x, int y, int size, SDL_Color color) {
    if (system->shapeCount < system->maxShapes) {
        system->shapes[system->shapeCount].type = SHAPE_SQUARE;
        system->shapes[system->shapeCount].rect = (SDL_Rect){x - size/2, y - size/2, size, size};
        system->shapes[system->shapeCount].color = color;
        system->shapeCount++;
    }
}

void addCircleToSystem(LidarSystem* system, int x, int y, int radius, SDL_Color color) {
    if (system->shapeCount < system->maxShapes) {
        system->shapes[system->shapeCount].type = SHAPE_CIRCLE;
        system->shapes[system->shapeCount].rect = (SDL_Rect){x, y, 0, 0};
        system->shapes[system->shapeCount].radius = radius;
        system->shapes[system->shapeCount].color = color;
        system->shapeCount++;
    }
}

void addLineToSystem(LidarSystem* system, int x1, int y1, int x2, int y2, SDL_Color color) {
    if (system->shapeCount < system->maxShapes) {
        system->shapes[system->shapeCount].type = SHAPE_LINE;
        system->shapes[system->shapeCount].start = (SDL_Point){x1, y1};
        system->shapes[system->shapeCount].end = (SDL_Point){x2, y2};
        system->shapes[system->shapeCount].color = color;
        system->shapeCount++;
    }
}

void setLidarPosition(LidarSystem* system, int x, int y) {
    if (x < 0) x = 0;
    if (x >= SPACE_WIDTH) x = SPACE_WIDTH - 1;
    if (y < 0) y = 0;
    if (y >= SPACE_HEIGHT) y = SPACE_HEIGHT - 1;
    
    system->lidarPosition.x = x;
    system->lidarPosition.y = y;
}

void setNoiseLevel(LidarSystem* system, double stdDev) {
    system->noiseStdDev = stdDev;
}

void updateRays(LidarSystem* system) {
    double max_dist = sqrt(SPACE_WIDTH * SPACE_WIDTH + SPACE_HEIGHT * SPACE_HEIGHT);
    
    for (int i = 0; i < system->rayCount; i++) {
        double angle = 2 * PI * i / system->rayCount;
        double dx = cos(angle);
        double dy = sin(angle);
        
        system->rays[i].origin = system->lidarPosition;
        system->rays[i].angle = angle;
        system->rays[i].true_distance = max_dist; // Default to max distance
        system->rays[i].measured_distance = max_dist;
        system->rays[i].color = (SDL_Color){255, 255, 0, 255};
        
        // Set end point to max distance initially
        system->rays[i].end_point.x = system->lidarPosition.x + dx * max_dist;
        system->rays[i].end_point.y = system->lidarPosition.y + dy * max_dist;
        
        // Check intersection with all shapes
        for (int j = 0; j < system->shapeCount; j++) {
            double distance;
            SDL_Point intersection;
            
            if (rayIntersectsShape(&system->rays[i], &system->shapes[j], &distance, &intersection)) {
                if (distance < system->rays[i].true_distance) {
                    system->rays[i].true_distance = distance;
                    system->rays[i].end_point = intersection;
                    
                    // Apply noise to the measured distance
                    if (system->noiseStdDev > 0.0) {
                        double noisy_distance = generateGaussianNoise(distance, system->noiseStdDev);
                        // Ensure measured distance is positive
                        system->rays[i].measured_distance = fmax(0.0, noisy_distance);
                        // Write the measured_point which is like end_point but with noise
                        system->rays[i].measured_point.x = system->lidarPosition.x + dx * system->rays[i].measured_distance;
                        system->rays[i].measured_point.y = system->lidarPosition.y + dy * system->rays[i].measured_distance;
                    } else {
                        system->rays[i].measured_distance = distance;
                    }
                }
            }
        }
        
        // If no intersection was found (ray hit nothing), still apply noise if needed
        if (system->rays[i].true_distance >= max_dist && system->noiseStdDev > 0.0) {
            // For rays that don't hit anything, we might still want to simulate noise
            // but typically these would return max range values
            system->rays[i].measured_distance = max_dist;
        }
    }
}

// The rest of the intersection functions remain the same as in your original code
bool rayIntersectsShape(Ray* ray, Shape* shape, double* distance, SDL_Point* intersection) {
    switch (shape->type) {
        case SHAPE_CIRCLE:
            return rayIntersectsCircle(ray, shape, distance, intersection);
        case SHAPE_SQUARE:
            return rayIntersectsSquare(ray, shape, distance, intersection);
        case SHAPE_LINE:
            return rayIntersectsLine(ray, shape, distance, intersection);
        default:
            return false;
    }
}


bool rayIntersectsCircle(Ray* ray, Shape* circle, double* distance, SDL_Point* intersection) {
    double cx = circle->rect.x;
    double cy = circle->rect.y;
    double r = circle->radius;
    
    double dx = cos(ray->angle);
    double dy = sin(ray->angle);
    
    double ox = ray->origin.x;
    double oy = ray->origin.y;
    
    double ocx = cx - ox;
    double ocy = cy - oy;
    
    double proj = ocx * dx + ocy * dy;
    
    double closestX = ox + dx * proj;
    double closestY = oy + dy * proj;
    
    double distToCenter = sqrt(pow(cx - closestX, 2) + pow(cy - closestY, 2));
    
    if (distToCenter > r) {
        return false;
    }
    
    double intersectionDist = proj - sqrt(r * r - distToCenter * distToCenter);
    
    if (intersectionDist < 0) {
        return false;
    }
    
    *distance = intersectionDist;
    intersection->x = ox + dx * intersectionDist;
    intersection->y = oy + dy * intersectionDist;
    
    return true;
}

bool rayIntersectsSquare(Ray* ray, Shape* square, double* distance, SDL_Point* intersection) {
    SDL_Rect rect = square->rect;
    double dx = cos(ray->angle);
    double dy = sin(ray->angle);
    double ox = ray->origin.x;
    double oy = ray->origin.y;
    
    double tMin = 0.0;
    double tMax = DBL_MAX;
    
    if (fabs(dx) < 1e-6) {
        if (ox < rect.x || ox > rect.x + rect.w) {
            return false;
        }
    } else {
        double t1 = (rect.x - ox) / dx;
        double t2 = (rect.x + rect.w - ox) / dx;
        
        if (t1 > t2) {
            double temp = t1;
            t1 = t2;
            t2 = temp;
        }
        
        if (t1 > tMin) tMin = t1;
        if (t2 < tMax) tMax = t2;
    }
    
    if (fabs(dy) < 1e-6) {
        if (oy < rect.y || oy > rect.y + rect.h) {
            return false;
        }
    } else {
        double t1 = (rect.y - oy) / dy;
        double t2 = (rect.y + rect.h - oy) / dy;
        
        if (t1 > t2) {
            double temp = t1;
            t1 = t2;
            t2 = temp;
        }
        
        if (t1 > tMin) tMin = t1;
        if (t2 < tMax) tMax = t2;
    }
    
    if (tMin > tMax || tMax < 0) {
        return false;
    }
    
    double t = tMin > 0 ? tMin : tMax;
    if (t < 0) {
        return false;
    }
    
    *distance = t;
    intersection->x = ox + dx * t;
    intersection->y = oy + dy * t;
    
    return true;
}

bool rayIntersectsLine(Ray* ray, Shape* lineShape, double* distance, SDL_Point* intersection) {
    SDL_Point p1 = lineShape->start;
    SDL_Point p2 = lineShape->end;
    
    double x1 = p1.x;
    double y1 = p1.y;
    double x2 = p2.x;
    double y2 = p2.y;
    
    double x3 = ray->origin.x;
    double y3 = ray->origin.y;
    double max_dist = sqrt(SPACE_WIDTH * SPACE_WIDTH + SPACE_HEIGHT * SPACE_HEIGHT);
    double x4 = ray->origin.x + cos(ray->angle) * max_dist;
    double y4 = ray->origin.y + sin(ray->angle) * max_dist;
    
    double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    
    if (fabs(denom) < 1e-6) {
        return false;
    }
    
    double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
    
    if (t >= 0 && t <= 1 && u >= 0) {
        *distance = u * sqrt(pow(x4 - x3, 2) + pow(y4 - y3, 2));
        intersection->x = x1 + t * (x2 - x1);
        intersection->y = y1 + t * (y2 - y1);
        return true;
    }
    
    return false;
}