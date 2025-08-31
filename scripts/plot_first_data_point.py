import numpy as np
import matplotlib.pyplot as plt
import struct

def read_lidar_data(filename):
    with open(filename, 'rb') as f:
        # Read metadata header
        metadata = f.read(12)
        if len(metadata) < 12:
            raise ValueError("File is too small to contain metadata")
            
        num_samples, ray_count, noise_level_int = struct.unpack('iii', metadata)
        noise_level = noise_level_int / 1000.0
        
        print(f"Metadata: {num_samples} samples, {ray_count} rays, noise level: {noise_level:.3f}")
        
        # Calculate sample size and verify file consistency
        expected_file_size = 12 + num_samples * (ray_count * 4 * 2 + 8)
        f.seek(0, 2)  # Seek to end of file
        actual_file_size = f.tell()
        f.seek(12)  # Seek back to start of data
        
        if actual_file_size != expected_file_size:
            print(f"Warning: File size mismatch! Expected: {expected_file_size}, Actual: {actual_file_size}")
            # Adjust num_samples based on actual file size
            actual_samples = (actual_file_size - 12) // (ray_count * 4 * 2 + 8)
            print(f"Using {actual_samples} samples instead of {num_samples}")
            num_samples = actual_samples
        
        # Read first sample only
        sample_size = ray_count * 4 * 2 + 8
        data = f.read(sample_size)
        
        if len(data) < sample_size:
            raise ValueError("File doesn't contain complete sample data")
    
    # Extract data from the sample
    true_dists = np.frombuffer(data[:ray_count*4], dtype=np.float32)
    measured_dists = np.frombuffer(data[ray_count*4:ray_count*8], dtype=np.float32)
    lidar_x, lidar_y = struct.unpack('ff', data[ray_count*8:ray_count*8+8])
    
    angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)
    
    print(f"Sample data: Lidar position ({lidar_x:.1f}, {lidar_y:.1f})")
    print(f"True distances range: [{true_dists.min():.1f}, {true_dists.max():.1f}]")
    print(f"Measured distances range: [{measured_dists.min():.1f}, {measured_dists.max():.1f}]")
    
    return angles, true_dists, measured_dists, lidar_x, lidar_y, noise_level

# Read data with proper metadata handling
try:
    angles, true_dists, measured_dists, lidar_x, lidar_y, noise_level = read_lidar_data('data/lidar_training_data.bin')
    
    # Create scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(angles, true_dists, color='blue', alpha=0.7, label='True Distance', s=10)
    plt.scatter(angles, measured_dists, color='red', alpha=0.7, label='Measured Distance', s=10)

    plt.xlabel('Angle (radians)')
    plt.ylabel('Distance')
    plt.title(f'Lidar Measurements: True vs Noisy (Noise Level: {noise_level:.1f})\nLidar Position: ({lidar_x:.1f}, {lidar_y:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print("Error: lidar_training_data.bin not found!")
except ValueError as e:
    print(f"Error reading file: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")