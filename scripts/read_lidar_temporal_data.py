import numpy as np
import matplotlib.pyplot as plt
import struct
import time
from matplotlib.animation import FuncAnimation

def read_lidar_temporal_data(filename):
    with open(filename, 'rb') as f:
        # Read metadata header (5 integers)
        metadata = f.read(20)  # 5 integers * 4 bytes each = 20 bytes
        if len(metadata) < 20:
            raise ValueError("File is too small to contain metadata")
            
        num_temporal_examples, steps_per_trajectory, ray_count, noise_level_int, sample_size = struct.unpack('iiiii', metadata)
        noise_level = noise_level_int / 1000.0
        
        print(f"Metadata: {num_temporal_examples} temporal examples, {steps_per_trajectory} steps, {ray_count} rays, noise level: {noise_level:.3f}")
        
        # Calculate expected file size and verify
        expected_file_size = 20 + num_temporal_examples * steps_per_trajectory * sample_size
        f.seek(0, 2)  # Seek to end of file
        actual_file_size = f.tell()
        f.seek(20)  # Seek back to start of data
        
        if actual_file_size != expected_file_size:
            print(f"Warning: File size mismatch! Expected: {expected_file_size}, Actual: {actual_file_size}")
        
        # Read all data
        all_data = []
        current_temporal_id = -1
        current_sequence = []
        
        for i in range(num_temporal_examples * steps_per_trajectory):
            data = f.read(sample_size)
            if len(data) < sample_size:
                break
                
            # Parse sample data
            true_dists = np.frombuffer(data[:ray_count*4], dtype=np.float32)
            measured_dists = np.frombuffer(data[ray_count*4:ray_count*8], dtype=np.float32)
            lidar_x, lidar_y, temporal_id, step_index = struct.unpack('ffii', data[ray_count*8:ray_count*8+16])
            
            sample = {
                'true_dists': true_dists,
                'measured_dists': measured_dists,
                'lidar_x': lidar_x,
                'lidar_y': lidar_y,
                'temporal_id': temporal_id,
                'step_index': step_index
            }
            
            # Group by temporal_id
            if temporal_id != current_temporal_id:
                if current_sequence:
                    all_data.append(current_sequence)
                current_sequence = []
                current_temporal_id = temporal_id
            
            current_sequence.append(sample)
        
        if current_sequence:
            all_data.append(current_sequence)
    
    angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)
    
    return angles, all_data, noise_level, num_temporal_examples, steps_per_trajectory

# Read temporal data
try:
    angles, temporal_data, noise_level, num_temporal_examples, steps_per_trajectory = read_lidar_temporal_data('data/lidar_training_data.bin')
    
    print(f"Loaded {len(temporal_data)} temporal sequences")
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Initialize plots
    scatter_true = ax1.scatter([], [], color='blue', alpha=0.7, label='True Distance', s=10)
    scatter_measured = ax1.scatter([], [], color='red', alpha=0.7, label='Measured Distance', s=10)
    
    # Position plot
    position_plot, = ax2.plot([], [], 'bo-', alpha=0.7, label='Lidar Trajectory')
    current_pos, = ax2.plot([], [], 'ro', markersize=8, label='Current Position')
    
    # Set up axes
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(0, max([max(sample['true_dists']) for sequence in temporal_data for sample in sequence]) * 1.1)
    ax1.set_xlabel('Angle (radians)')
    ax1.set_ylabel('Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, 800)  # Assuming SPACE_WIDTH
    ax2.set_ylim(0, 600)  # Assuming SPACE_HEIGHT
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Lidar Movement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Animation variables
    current_temporal_idx = 0
    current_step_idx = 0
    frame_count = [0]  # Use list to make it mutable in animation function
    
    def update(frame):
        
        # Get current sample
        current_sequence = temporal_data[current_temporal_idx]
        sample = current_sequence[current_step_idx]
        
        # Update scatter plots
        scatter_true.set_offsets(np.column_stack([angles, sample['true_dists']]))
        scatter_measured.set_offsets(np.column_stack([angles, sample['measured_dists']]))
        
        # Update position plot - show entire trajectory
        x_positions = [s['lidar_x'] for s in current_sequence]
        y_positions = [s['lidar_y'] for s in current_sequence]
        position_plot.set_data(x_positions, y_positions)
        current_pos.set_data([sample['lidar_x']], [sample['lidar_y']])
        
        # Update titles
        ax1.set_title(f'Temporal Example {current_temporal_idx}, Step {current_step_idx}/{steps_per_trajectory-1}\n'
                     f'Position: ({sample["lidar_x"]:.1f}, {sample["lidar_y"]:.1f})')
        
        # Move to next step
        current_step_idx += 1
        if current_step_idx >= steps_per_trajectory:
            current_step_idx = 0
            current_temporal_idx = (current_temporal_idx + 1) % num_temporal_examples
        
        frame_count[0] += 1
        if frame_count[0] % 10 == 0:
            print(f"Displaying temporal example {current_temporal_idx}, step {current_step_idx}")
        
        return scatter_true, scatter_measured, position_plot, current_pos
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=num_temporal_examples * steps_per_trajectory, 
                       interval=50, blit=True)  # 50ms = 20fps
    
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print("Error: lidar_training_data.bin not found!")
    print("Please run the C program first to generate the temporal data.")
except ValueError as e:
    print(f"Error reading file: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")