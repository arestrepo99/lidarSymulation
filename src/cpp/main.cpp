#include <iostream>
#include <vector>
#include <iomanip>
#include "lidar_denoiser.h"

int main() {
    std::cout << "Lidar Denoiser C++ Test" << std::endl;
    std::cout << "=======================" << std::endl;
    
    try {
        // Create denoiser
        LidarDenoiser denoiser;
        
        // Load model
        if (!denoiser.load_model("lidar_denoiser_traced.pt")) {
            std::cerr << "Failed to load model!" << std::endl;
            return 1;
        }
        
        // Predict from data file
        std::vector<float> result = denoiser.predict_from_file("lidar_training_data.bin");
        
        std::cout << "\nPrediction completed successfully!" << std::endl;
        std::cout << "Output size: " << result.size() << std::endl;
        
        // Print first 10 values
        std::cout << "\nFirst 10 predicted values:" << std::endl;
        for (int i = 0; i < 10 && i < result.size(); ++i) {
            std::cout << "[" << i << "]: " << std::fixed << std::setprecision(2) 
                     << result[i] << std::endl;
        }
        
        // Print statistics
        float min_val = *std::min_element(result.begin(), result.end());
        float max_val = *std::max_element(result.begin(), result.end());
        float avg_val = std::accumulate(result.begin(), result.end(), 0.0f) / result.size();
        
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "Min: " << min_val << std::endl;
        std::cout << "Max: " << max_val << std::endl;
        std::cout << "Avg: " << avg_val << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}