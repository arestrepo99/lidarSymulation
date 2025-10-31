# 2D LIDAR Simulation with Deep Learning Denoising

A comprehensive simulation of a 2D LIDAR system with integrated deep learning-based denoising capabilities. This project combines C-based LIDAR simulation with PyTorch-powered neural networks for real-time noise reduction in LIDAR data.

## 📋 Overview

This project implements:
- **2D LIDAR Simulation**: Realistic LIDAR data generation written in C
- **Deep Learning Denoising**: Neural network models for noise reduction  
- **Temporal Processing**: LSTM-based models that leverage historical data
- **Real-time Visualization**: SDL2-based rendering of LIDAR scans

The denoising approach is inspired by the research paper: ["A Novel Lidar Signal Denoising Method Based on Convolutional Autoencoding Deep Learning Neural Network"](https://www.mdpi.com/2073-4433/12/11/1403) from Atmosphere MDPI journal.

## 🏗️ Architecture

### Core Components

1. **LIDAR Simulation (C)**
   - Synthetic LIDAR data generation
   - Ray-casting for obstacle detection
   - Configurable noise parameters

2. **Neural Network Models (PyTorch/C++)**
   - **1D CNN**: Alternative spatial processing approach
   - **1D CNN + LSTM**: Hybrid architecture for temporal denoising
   - Real-time inference integration

3. **Data Pipeline**
   - Training data generation
   - Temporal sequence handling
   - Preprocessing and normalization

## 🚀 Quick Start

### Prerequisites

- GCC/Clang compiler
- SDL2 development libraries  
- PyTorch C++ API (LibTorch)
- Raylib (for visualization)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/lidar-simulation.git
cd lidar-simulation

# Install dependencies (macOS)
make install-deps

# Build the project
make all
```

### Usage

```bash
# Run the main LIDAR simulation with denoising
make run-simulation

# Generate training data
make run-data-gen

# Generate temporal training data for LSTM models
make run-data-gen-temporal

# Run C integration tests
make run-test
```

## 🎯 Key Features

### LIDAR Simulation
- Configurable angular resolution (360° scans)
- Adjustable maximum range detection
- Realistic noise modeling (G
