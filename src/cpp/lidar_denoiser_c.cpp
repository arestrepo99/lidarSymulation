#include "lidar_denoiser_c.h"
#include "lidar_denoiser.h"
#include <string>
#include <vector>

extern "C" {

lidar_denoiser_handle lidar_denoiser_create() {
    return new LidarDenoiser();
}

void lidar_denoiser_destroy(lidar_denoiser_handle handle) {
    if (handle) {
        delete static_cast<LidarDenoiser*>(handle);
    }
}

int lidar_denoiser_load_model(lidar_denoiser_handle handle, const char* model_path) {
    if (!handle) return 0;
    
    LidarDenoiser* denoiser = static_cast<LidarDenoiser*>(handle);
    return denoiser->load_model(model_path) ? 1 : 0;
}

int lidar_denoiser_predict(lidar_denoiser_handle handle, 
                          const float* input, int input_size,
                          float* output, int output_size) {
    if (!handle || !input || !output) return 0;
    
    try {
        LidarDenoiser* denoiser = static_cast<LidarDenoiser*>(handle);
        std::vector<float> input_vec(input, input + input_size);
        std::vector<float> result = denoiser->predict(input_vec);
        
        if (output_size >= static_cast<int>(result.size())) {
            std::copy(result.begin(), result.end(), output);
            return 1;
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

int lidar_denoiser_predict_from_file(lidar_denoiser_handle handle, 
                                    const char* data_file,
                                    float* output, int output_size) {
    if (!handle || !data_file || !output) return 0;
    
    LidarDenoiser* denoiser = static_cast<LidarDenoiser*>(handle);
    std::vector<float> result = denoiser->predict_from_file(data_file);
    
    if (output_size >= static_cast<int>(result.size())) {
        std::copy(result.begin(), result.end(), output);
        return 1;
    }
    return 0;
}

} // extern "C"