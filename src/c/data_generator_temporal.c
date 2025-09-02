#include "lidar_system.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#define NUM_TEMPORAL_EXAMPLES 30000
#define STEPS_PER_TRAJECTORY 20
#define MAX_SHAPES_PER_SAMPLE 10
#define RAY_COUNT 120
#define NOISE_LEVEL 10.0

typedef struct {
    float true_distances[RAY_COUNT];
    float measured_distances[RAY_COUNT];
    float lidar_x;
    float lidar_y;
    int temporal_id;  // Identifier for the temporal sequence
    int step_index;   // Step within the temporal sequence (0 to STEPS_PER_TRAJECTORY-1)
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

void generateRandomLidarPosition(LidarSystem* system, float* x, float* y) {
    *x = (float)(rand() % (SPACE_WIDTH - 100) + 50);
    *y = (float)(rand() % (SPACE_HEIGHT - 100) + 50);
    setLidarPosition(system, (int)*x, (int)*y);
}

void interpolateLidarPosition(LidarSystem* system, float start_x, float start_y, 
                             float end_x, float end_y, int step, int total_steps) {
    float t = (float)step / (float)(total_steps - 1);
    float current_x = start_x + (end_x - start_x) * t;
    float current_y = start_y + (end_y - start_y) * t;
    setLidarPosition(system, (int)current_x, (int)current_y);
}

void writeSampleToFile(FILE* file, LidarSystem* system, int temporal_id, int step_index) {
    SampleData sample;
    
    // Store lidar position
    sample.lidar_x = (float)system->lidarPosition.x;
    sample.lidar_y = (float)system->lidarPosition.y;
    
    // Store temporal information
    sample.temporal_id = temporal_id;
    sample.step_index = step_index;
    
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
    FILE* output_file = fopen("data/lidar_temporal_data.bin", "wb");
    if (!output_file) {
        printf("Error opening output file!\n");
        destroyLidarSystem(lidarSystem);
        return 1;
    }
    
    // Write header with metadata
    int metadata[5] = {
        NUM_TEMPORAL_EXAMPLES, 
        STEPS_PER_TRAJECTORY,
        RAY_COUNT, 
        (int)(NOISE_LEVEL * 1000),
        (int)sizeof(SampleData)
    };
    fwrite(metadata, sizeof(int), 5, output_file);
    
    int total_samples = NUM_TEMPORAL_EXAMPLES * STEPS_PER_TRAJECTORY;
    printf("Generating %d temporal examples with %d steps each (%d total samples)...\n", 
           NUM_TEMPORAL_EXAMPLES, STEPS_PER_TRAJECTORY, total_samples);
    
    for (int temporal_id = 0; temporal_id < NUM_TEMPORAL_EXAMPLES; temporal_id++) {
        if (temporal_id % 1000 == 0) {
            printf("Generated %d temporal examples...\n", temporal_id);
        }
        
        // Generate new random environment for this temporal sequence
        generateRandomEnvironment(lidarSystem);
        
        // Generate start and end positions for the trajectory
        float start_x, start_y, end_x, end_y;
        generateRandomLidarPosition(lidarSystem, &start_x, &start_y);
        generateRandomLidarPosition(lidarSystem, &end_x, &end_y);
        
        // Generate all steps in the trajectory
        for (int step = 0; step < STEPS_PER_TRAJECTORY; step++) {
            // Interpolate lidar position along the trajectory
            interpolateLidarPosition(lidarSystem, start_x, start_y, end_x, end_y, 
                                   step, STEPS_PER_TRAJECTORY);
            
            // Update ray measurements
            updateRays(lidarSystem);
            
            // Write sample to file with temporal information
            writeSampleToFile(output_file, lidarSystem, temporal_id, step);
        }
    }
    
    fclose(output_file);
    destroyLidarSystem(lidarSystem);
    
    printf("Data generation complete! Generated %d temporal examples.\n", NUM_TEMPORAL_EXAMPLES);
    printf("File format: %d temporal sequences, %d steps per sequence\n", 
           NUM_TEMPORAL_EXAMPLES, STEPS_PER_TRAJECTORY);
    printf("Each sample contains:\n");
    printf("  - %d true distances (float)\n", RAY_COUNT);
    printf("  - %d measured distances (float)\n", RAY_COUNT);
    printf("  - Lidar X position (float)\n");
    printf("  - Lidar Y position (float)\n");
    printf("  - Temporal ID (int)\n");
    printf("  - Step index (int)\n");
    printf("Total file size: %.2f MB\n", 
           (total_samples * sizeof(SampleData) + sizeof(int) * 5) / (1024.0 * 1024.0));
    
    return 0;
}