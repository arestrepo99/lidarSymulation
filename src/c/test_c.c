#include <stdio.h>
#include <stdlib.h>
#include "lidar_denoiser_c.h"

int main() {
    printf("Lidar Denoiser C Test\n");
    printf("=====================\n");
    
    // Create denoiser
    lidar_denoiser_handle handle = lidar_denoiser_create();
    if (!handle) {
        printf("Failed to create denoiser!\n");
        return 1;
    }
    
    // Load model
    if (!lidar_denoiser_load_model(handle, "models/lidar_denoiser_traced.pt")) {
        printf("Failed to load model!\n");
        lidar_denoiser_destroy(handle);
        return 1;
    }
    
    // Prepare output buffer
    const int output_size = 60;
    float* output = (float*)malloc(output_size * sizeof(float));
    if (!output) {
        printf("Memory allocation failed!\n");
        lidar_denoiser_destroy(handle);
        return 1;
    }
    
    // Predict from file
    if (lidar_denoiser_predict_from_file(handle, "data/lidar_training_data.bin", output, output_size)) {
        printf("Prediction completed successfully!\n");
        
        // Print first 10 values
        printf("\nFirst 10 predicted values:\n");
        for (int i = 0; i < 10; i++) {
            printf("[%d]: %.2f\n", i, output[i]);
        }
    } else {
        printf("Prediction failed!\n");
    }
    
    // Cleanup
    free(output);
    lidar_denoiser_destroy(handle);
    
    return 0;
}