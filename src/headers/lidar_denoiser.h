#ifndef LIDAR_DENOISER_H
#define LIDAR_DENOISER_H

#include <torch/script.h>
#include <vector>
#include <string>

class LidarDenoiser {
public:
    LidarDenoiser();
    ~LidarDenoiser();
    
    bool load_model(const std::string& model_path);
    std::vector<float> predict(const std::vector<float>& input);
    std::vector<float> predict_from_file(const std::string& data_file);
    
private:
    torch::jit::script::Module model;
    bool model_loaded;
    const int input_size = 120;
    const float max_distance = 2000.0f;
};

#endif // LIDAR_DENOISER_H