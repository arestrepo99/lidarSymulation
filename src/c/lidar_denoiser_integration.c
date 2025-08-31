#include "lidar_denoiser_integration.h"
#include "lidar_system.h"
#include "lidar_denoiser_c.h"
#include <stdio.h>
#include <string.h>

// Global denoiser handle (could be made thread-safe if needed)
static lidar_denoiser_handle denoiser_handle = NULL;
static bool denoiser_loaded = false;

bool init_lidar_denoiser(const char* model_path) {
    // Create denoiser instance
    denoiser_handle = lidar_denoiser_create();
    if (!denoiser_handle) {
        fprintf(stderr, "Failed to create lidar denoiser\n");
        return false;
    }
    
    // Load the model
    if (!lidar_denoiser_load_model(denoiser_handle, model_path)) {
        fprintf(stderr, "Failed to load lidar denoiser model from: %s\n", model_path);
        lidar_denoiser_destroy(denoiser_handle);
        denoiser_handle = NULL;
        return false;
    }
    
    denoiser_loaded = true;
    printf("Lidar denoiser initialized successfully with model: %s\n", model_path);
    return true;
}

void cleanup_lidar_denoiser() {
    if (denoiser_handle) {
        lidar_denoiser_destroy(denoiser_handle);
        denoiser_handle = NULL;
    }
    denoiser_loaded = false;
}

bool denoiseRays(LidarSystem* system) {
    if (!system || !denoiser_loaded) {
        return false;
    }
    
    // Prepare input data (measured distances)
    float* input_data = (float*)malloc(system->rayCount * sizeof(float));
    if (!input_data) {
        fprintf(stderr, "Memory allocation failed for denoiser input\n");
        return false;
    }
    
    // Copy measured distances to input array
    for (int i = 0; i < system->rayCount; i++) {
        input_data[i] = (float)system->rays[i].measured_distance;
    }
    
    // Prepare output buffer
    float* output_data = (float*)malloc(system->rayCount * sizeof(float));
    if (!output_data) {
        fprintf(stderr, "Memory allocation failed for denoiser output\n");
        free(input_data);
        return false;
    }
    
    // Run denoising prediction
    int success = lidar_denoiser_predict(
        denoiser_handle,
        input_data,
        system->rayCount,
        output_data,
        system->rayCount
    );
    
    if (success) {
        // Update the rays with denoised distances
        for (int i = 0; i < system->rayCount; i++) {
            system->rays[i].predicted_distance = (double)output_data[i];
            
            // Also update the end point for rendering if needed
            if (system->render) {
                double angle = system->rays[i].angle;
                double dx = cos(angle);
                double dy = sin(angle);
                
                system->rays[i].predicted_point.x = system->lidarPosition.x + dx * output_data[i];
                system->rays[i].predicted_point.y = system->lidarPosition.y + dy * output_data[i];
            }
        }
        
        // printf("Successfully denoised %d rays\n", system->rayCount);
    } else {
        fprintf(stderr, "Denoising prediction failed\n");
    }
    
    // Cleanup
    free(input_data);
    free(output_data);
    
    return success;
}

bool denoiseRaysFromFile(LidarSystem* system, const char* data_file) {
    if (!system || !denoiser_loaded || !data_file) {
        return false;
    }
    
    // Prepare output buffer
    float* output_data = (float*)malloc(system->rayCount * sizeof(float));
    if (!output_data) {
        fprintf(stderr, "Memory allocation failed for denoiser output\n");
        return false;
    }
    
    // Run denoising prediction from file
    int success = lidar_denoiser_predict_from_file(
        denoiser_handle,
        data_file,
        output_data,
        system->rayCount
    );
    
    if (success) {
        // Update the rays with denoised distances
        for (int i = 0; i < system->rayCount; i++) {
            system->rays[i].measured_distance = (double)output_data[i];
            
            // Also update the end point for rendering if needed
            if (system->render) {
                double angle = system->rays[i].angle;
                double dx = cos(angle);
                double dy = sin(angle);
                
                system->rays[i].end_point.x = system->lidarPosition.x + dx * output_data[i];
                system->rays[i].end_point.y = system->lidarPosition.y + dy * output_data[i];
            }
        }
        
        printf("Successfully denoised rays from file: %s\n", data_file);
    } else {
        fprintf(stderr, "Denoising prediction from file failed\n");
    }
    
    free(output_data);
    return success;
}