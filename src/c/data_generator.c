#include "lidar_system.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#define NUM_SAMPLES 200000
#define MAX_SHAPES_PER_SAMPLE 10
#define RAY_COUNT 120
#define NOISE_LEVEL 10.0

typedef struct {
    float true_distances[RAY_COUNT];
    float measured_distances[RAY_COUNT];
    float lidar_x;
    float lidar_y;
} SampleData;

void generateRandomEnvironment(LidarSystem* system) {
    // Clear existing shapes
    system->shapeCount = 0;
    
    // Add boundary walls (always present)
    addLineToSystem(system, 0, 0, SPACE_WIDTH, 0, (SDL_Color){255, 255, 255, 255});
    addLineToSystem(system, SPACE_WIDTH, 0, SPACE_WIDTH, SPACE_HEIGHT, (SDL_Color){255, 255, 255, 255});
    addLineToSystem(system, SPACE_WIDTH, SPACE_HEIGHT, 0, SPACE_HEIGHT, (SDL_Color){255, 255, 255, 255});
    addLineToSystem(system, 0, SPACE_HEIGHT, 0, 0, (SDL_Color){255, 255, 255, 255});
    
    // Generate random number of shapes (1 to MAX_SHAPES_PER_SAMPLE)
    int num_shapes = rand() % MAX_SHAPES_PER_SAMPLE + 1;
    
    for (int i = 0; i < num_shapes; i++) {
        int shape_type = rand() % 3;
        SDL_Color color = {rand() % 256, rand() % 256, rand() % 256, 255};
        
        switch (shape_type) {
            case 0: { // Square
                int x = rand() % (SPACE_WIDTH - 100) + 50;
                int y = rand() % (SPACE_HEIGHT - 100) + 50;
                int size = rand() % 80 + 20;
                addSquareToSystem(system, x, y, size, color);
                break;
            }
            case 1: { // Circle
                int x = rand() % (SPACE_WIDTH - 100) + 50;
                int y = rand() % (SPACE_HEIGHT - 100) + 50;
                int radius = rand() % 50 + 15;
                addCircleToSystem(system, x, y, radius, color);
                break;
            }
            case 2: { // Line
                int x1 = rand() % SPACE_WIDTH;
                int y1 = rand() % SPACE_HEIGHT;
                int x2 = rand() % SPACE_WIDTH;
                int y2 = rand() % SPACE_HEIGHT;
                addLineToSystem(system, x1, y1, x2, y2, color);
                break;
            }
        }
    }
}

void generateRandomLidarPosition(LidarSystem* system) {
    int x = rand() % (SPACE_WIDTH - 100) + 50;
    int y = rand() % (SPACE_HEIGHT - 100) + 50;
    setLidarPosition(system, x, y);
}

void writeSampleToFile(FILE* file, LidarSystem* system) {
    SampleData sample;
    
    // Store lidar position
    sample.lidar_x = (float)system->lidarPosition.x;
    sample.lidar_y = (float)system->lidarPosition.y;
    
    // Store distance data
    for (int i = 0; i < RAY_COUNT; i++) {
        sample.true_distances[i] = (float)system->rays[i].true_distance;
        sample.measured_distances[i] = (float)system->rays[i].measured_distance;
    }
    
    // Write to binary file
    fwrite(&sample, sizeof(SampleData), 1, file);
}

int main() {
    srand(time(NULL));
    
    // Create lidar system with rendering disabled
    LidarSystem* lidarSystem = createLidarSystem(MAX_SHAPES_PER_SAMPLE + 4, RAY_COUNT, false);
    setNoiseLevel(lidarSystem, NOISE_LEVEL);
    
    // Make directory data if not exists
    mkdir("data", 0755);

    // Open output file
    FILE* output_file = fopen("data/lidar_training_data.bin", "wb");
    if (!output_file) {
        printf("Error opening output file!\n");
        destroyLidarSystem(lidarSystem);
        return 1;
    }
    
    // Write header with metadata
    int metadata[3] = {NUM_SAMPLES, RAY_COUNT, (int)(NOISE_LEVEL * 1000)};
    fwrite(metadata, sizeof(int), 3, output_file);
    
    printf("Generating %d samples...\n", NUM_SAMPLES);
    
    for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
        if (sample_idx % 1000 == 0) {
            printf("Generated %d samples...\n", sample_idx);
        }
        
        // Generate new random environment
        generateRandomEnvironment(lidarSystem);
        
        // Generate multiple lidar positions for the same environment
        int positions_per_environment = 5;
        for (int pos_idx = 0; pos_idx < positions_per_environment; pos_idx++) {
            if (sample_idx * positions_per_environment + pos_idx >= NUM_SAMPLES) {
                break;
            }
            
            generateRandomLidarPosition(lidarSystem);
            updateRays(lidarSystem);
            writeSampleToFile(output_file, lidarSystem);
        }
    }
    
    fclose(output_file);
    destroyLidarSystem(lidarSystem);
    
    printf("Data generation complete! Generated %d samples.\n", NUM_SAMPLES);
    printf("File format: %d samples, %d rays per sample, noise level: %.1f\n", 
           NUM_SAMPLES, RAY_COUNT, NOISE_LEVEL);
    printf("Each sample contains:\n");
    printf("  - %d true distances (float)\n", RAY_COUNT);
    printf("  - %d measured distances (float)\n", RAY_COUNT);
    printf("  - Lidar X position (float)\n");
    printf("  - Lidar Y position (float)\n");
    printf("Total file size: %.2f MB\n", 
           (NUM_SAMPLES * sizeof(SampleData) + sizeof(int) * 3) / (1024.0 * 1024.0));
    
    return 0;
}