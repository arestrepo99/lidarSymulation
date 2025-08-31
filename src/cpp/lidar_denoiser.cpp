#include "lidar_denoiser.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
// #include <torch/torch.h>

LidarDenoiser::LidarDenoiser() : model_loaded(false) {
    // Initialize Torch
    torch::manual_seed(42);
}

LidarDenoiser::~LidarDenoiser() {
    // Cleanup
}

bool LidarDenoiser::load_model(const std::string& model_path) {
    try {
        // Load the TorchScript model
        model = torch::jit::load(model_path);
        model.eval();
        model_loaded = true;
        std::cout << "Model loaded successfully from: " << model_path << std::endl;
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> LidarDenoiser::predict(const std::vector<float>& input) {
    if (!model_loaded) {
        throw std::runtime_error("Model not loaded. Call load_model() first.");
    }
    
    if (input.size() != input_size) {
        throw std::runtime_error("Input size mismatch. Expected " + 
                                std::to_string(input_size) + " elements.");
    }
    
    try {
        // Convert input to tensor
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<float*>(input.data()), 
            {1, 1, input_size}, 
            torch::kFloat32
        ).clone();  // Clone to ensure ownership
        
        // Normalize input (0-1 range)
        input_tensor = input_tensor / max_distance;
        
        // Create input vector
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Run prediction
        torch::Tensor output_tensor = model.forward(inputs).toTensor();
        
        // Denormalize output
        output_tensor = output_tensor * max_distance;
        
        // Convert to vector
        std::vector<float> output(output_tensor.data_ptr<float>(),
                                 output_tensor.data_ptr<float>() + output_tensor.numel());
        
        return output;
        
    } catch (const c10::Error& e) {
        throw std::runtime_error("Prediction error: " + std::string(e.what()));
    }
}

std::vector<float> LidarDenoiser::predict_from_file(const std::string& data_file) {
    std::ifstream file(data_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open data file: " + data_file);
    }
    
    // Read metadata (first 12 bytes)
    int num_samples, ray_count, noise_level_int;
    file.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
    file.read(reinterpret_cast<char*>(&ray_count), sizeof(int));
    file.read(reinterpret_cast<char*>(&noise_level_int), sizeof(int));
    
    std::cout << "Dataset info: " << num_samples << " samples, " 
              << ray_count << " rays, noise level: " 
              << (noise_level_int / 1000.0f) << std::endl;
    
    if (ray_count != input_size) {
        throw std::runtime_error("Ray count mismatch. Expected " + 
                                std::to_string(input_size) + ", got " + 
                                std::to_string(ray_count));
    }
    
    // Read the first sample
    const int sample_size = ray_count * 4 * 2 + 8;
    file.seekg(12); // Skip metadata
    
    std::vector<char> sample_data(sample_size);
    file.read(sample_data.data(), sample_size);
    
    if (file.gcount() != sample_size) {
        throw std::runtime_error("Incomplete sample data");
    }
    
    // Extract measured distances (second half of sample)
    std::vector<float> measured_dists(ray_count);
    std::memcpy(measured_dists.data(), 
               sample_data.data() + ray_count * 4, 
               ray_count * sizeof(float));
    
    std::cout << "First sample loaded. Range: [" 
              << *std::min_element(measured_dists.begin(), measured_dists.end())
              << ", "
              << *std::max_element(measured_dists.begin(), measured_dists.end())
              << "]" << std::endl;
    
    return predict(measured_dists);
}