#ifndef LIDAR_DENOISER_C_H
#define LIDAR_DENOISER_C_H

#ifdef __cplusplus
extern "C" {
#endif

// C interface for the LidarDenoiser
typedef void* lidar_denoiser_handle;

// Create a new denoiser instance
lidar_denoiser_handle lidar_denoiser_create();

// Destroy denoiser instance
void lidar_denoiser_destroy(lidar_denoiser_handle handle);

// Load model from file
int lidar_denoiser_load_model(lidar_denoiser_handle handle, const char* model_path);

// Predict from array
int lidar_denoiser_predict(lidar_denoiser_handle handle, 
                          const float* input, int input_size,
                          float* output, int output_size);

// Predict from data file
int lidar_denoiser_predict_from_file(lidar_denoiser_handle handle, 
                                    const char* data_file,
                                    float* output, int output_size);

#ifdef __cplusplus
}
#endif

#endif // LIDAR_DENOISER_C_H