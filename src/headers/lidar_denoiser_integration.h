#ifndef LIDAR_DENOISER_INTEGRATION_H
#define LIDAR_DENOISER_INTEGRATION_H

#include <stdbool.h>
#include "lidar_system.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the lidar denoiser with a model file
 * 
 * @param model_path Path to the trained model file
 * @return true if initialization successful, false otherwise
 */
bool init_lidar_denoiser(const char* model_path);

/**
 * @brief Clean up the lidar denoiser resources
 */
void cleanup_lidar_denoiser();

/**
 * @brief Denoise the rays in the lidar system using the loaded model
 * 
 * @param system Pointer to the LidarSystem containing rays to denoise
 * @return true if denoising successful, false otherwise
 */
bool denoiseRays(LidarSystem* system);

/**
 * @brief Denoise rays using data from a file
 * 
 * @param system Pointer to the LidarSystem
 * @param data_file Path to the data file containing ray measurements
 * @return true if denoising successful, false otherwise
 */
bool denoiseRaysFromFile(LidarSystem* system, const char* data_file);

#ifdef __cplusplus
}
#endif

#endif // LIDAR_DENOISER_INTEGRATION_H