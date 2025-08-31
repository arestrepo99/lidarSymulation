import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import time

# Set device for Apple Silicon GPU (MPS) if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the Denoising Model
class LidarDenoiserConv(nn.Module):
    def __init__(self, input_size=60):
        super(LidarDenoiserConv, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 2**5, kernel_size=5, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(2**5, 2**6, kernel_size=5, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(2**6, 2**7, kernel_size=5, padding_mode='circular'),
            nn.ReLU(),
        )
        # Calculate the size of the flattened vector
        self.conv_output_size = 128 * input_size # (channels * length)

        # Define your dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512), # Bottleneck layer
            nn.ReLU(),
            nn.Linear(512, self.conv_output_size), # Expand back to original size
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2**7, 2**6, kernel_size=5, padding_mode='circular'),
            nn.ReLU(),
            nn.ConvTranspose1d(2**6, 2**5, kernel_size=5, padding_mode='circular'),
            nn.ReLU(),
            nn.ConvTranspose1d(2**5, 1, kernel_size=5, padding_mode='circular'),
            nn.Sigmoid()  # Output between 0-1 (normalized distances)
        )
    
    def forward(self, x):
        # 1. Encode
        encoded = self.encoder(x) # shape: (batch_size, 128, L)
        
        # 2. Flatten for dense layers
        batch_size = encoded.shape[0]
        flattened = encoded.view(batch_size, -1) # shape: (batch_size, 128 * L)
        
        # 3. Process with dense layers
        dense_output = self.dense_layers(flattened) # shape: (batch_size, 128 * L)
        
        # 4. Reshape back to 3D for decoder
        reshaped = dense_output.view(batch_size, 128, self.input_size) # shape: (batch_size, 128, L)

        # 5. Decode
        decoded = self.decoder(reshaped) # shape: (batch_size, 1, L)
        
        return decoded

# Dataset class with proper metadata verification
class LidarDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            # Read metadata header
            metadata = f.read(12)
            if len(metadata) < 12:
                raise ValueError("File is too small to contain metadata")
                
            self.num_samples, self.ray_count, noise_level_int = struct.unpack('iii', metadata)
            self.noise_level = noise_level_int / 1000.0
            
            print(f"Metadata: {self.num_samples} samples, {self.ray_count} rays, noise level: {self.noise_level:.3f}")
            
            # Calculate expected file size and verify
            self.sample_size = self.ray_count * 4 * 2 + 8
            expected_file_size = 12 + self.num_samples * self.sample_size
            
            f.seek(0, 2)  # Seek to end of file
            actual_file_size = f.tell()
            f.seek(12)  # Seek back to start of data
            
            if actual_file_size != expected_file_size:
                print(f"Warning: File size mismatch! Expected: {expected_file_size}, Actual: {actual_file_size}")
                # Recalculate number of samples based on actual file size
                self.num_samples = (actual_file_size - 12) // self.sample_size
                print(f"Using {self.num_samples} samples based on actual file size")
            
            # Read all data at once for efficiency
            f.seek(12)
            self.raw_data = f.read()
        
        # Normalization parameters
        self.max_distance = 2000.0  # Maximum expected distance
        
        # Verify the first few samples to ensure consistency
        self._verify_sample_consistency()
    
    def _verify_sample_consistency(self):
        """Verify that all samples have the correct size"""
        print("Verifying sample consistency...")
        for i in range(min(10, self.num_samples)):  # Check first 10 samples
            start = i * self.sample_size
            end = start + self.sample_size
            
            if end > len(self.raw_data):
                print(f"Sample {i}: Incomplete data")
                continue
                
            sample_data = self.raw_data[start:end]
            
            # Check if we can properly parse the sample
            try:
                measured_dists = np.frombuffer(sample_data[self.ray_count*4:self.ray_count*8], dtype=np.float32)
                true_dists = np.frombuffer(sample_data[:self.ray_count*4], dtype=np.float32)
                
                if len(measured_dists) != self.ray_count or len(true_dists) != self.ray_count:
                    print(f"Sample {i}: Ray count mismatch! Expected {self.ray_count}, got {len(measured_dists)}/{len(true_dists)}")
                else:
                    print(f"Sample {i}: OK - True: [{true_dists.min():.1f}, {true_dists.max():.1f}], "
                          f"Measured: [{measured_dists.min():.1f}, {measured_dists.max():.1f}]")
                    
            except Exception as e:
                print(f"Sample {i}: Error parsing - {e}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.sample_size
        end = start + self.sample_size
        
        if end > len(self.raw_data):
            raise IndexError(f"Sample {idx} is out of bounds")
        
        sample_data = self.raw_data[start:end]
        
        # Extract data with proper copying to make tensors writable
        measured_dists = np.frombuffer(sample_data[self.ray_count*4:self.ray_count*8], dtype=np.float32).copy()
        true_dists = np.frombuffer(sample_data[:self.ray_count*4], dtype=np.float32).copy()
        
        # Verify ray count consistency
        if len(measured_dists) != self.ray_count or len(true_dists) != self.ray_count:
            # Pad or truncate to ensure consistent size
            measured_dists = np.resize(measured_dists, self.ray_count)
            true_dists = np.resize(true_dists, self.ray_count)
        
        # Normalize to 0-1 range and convert to tensors
        measured_normalized = torch.FloatTensor(measured_dists / self.max_distance)
        true_normalized = torch.FloatTensor(true_dists / self.max_distance)
        
        return measured_normalized.unsqueeze(0), true_normalized.unsqueeze(0)

# Training function with validation
def train_model():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    patience = 5  # Early stopping patience
    
    # Load dataset
    try:
        full_dataset = LidarDataset('data/lidar_training_data.bin')
        
        # Split dataset into train, validation, and test (60%, 20%, 20%)
        train_size = int(0.6 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        # Initialize model and move to device
        model = LidarDenoiserConv(input_size=full_dataset.ray_count).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"Starting training with {len(train_dataset)} samples, {full_dataset.ray_count} rays per sample")
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (noisy, clean) in enumerate(train_loader):
                # Move data to device
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    
                    outputs = model(noisy)
                    loss = criterion(outputs, clean)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            epoch_time = time.time() - start_time
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model (on CPU for compatibility)
                torch.save(model.cpu().state_dict(), 'models/lidar_denoiser_best.pth')
                model = model.to(device)  # Move back to device for continued training
            else:
                patience_counter += 1
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.1f}s, '
                  f'Train Loss: {avg_train_loss:.7f}, Val Loss: {avg_val_loss:.7f}, '
                  f'Patience: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        # Load best model (ensure it's on CPU for export compatibility)
        model.load_state_dict(torch.load('models/lidar_denoiser_best.pth', map_location='cpu'))
        model = model.to(device)  # Move to device for final evaluation
        
        # Final test evaluation
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f'Final Test Loss: {avg_test_loss:.6f}')
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Save final model (on CPU for compatibility)
        torch.save(model.cpu().state_dict(), 'models/lidar_denoiser_final.pth')
        model = model.to(device)  # Move back to device if needed
        print("Model saved as models/lidar_denoiser_final.pth")
        
        return model, test_loader, full_dataset
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Test the model
def test_model(model, test_loader, dataset):
    if model is None:
        return
        
    model.eval()
    
    # Get a test sample from the test loader
    test_iter = iter(test_loader)
    noisy, clean = next(test_iter)
    
    # Move to device for prediction
    noisy_device = noisy.to(device)
    
    with torch.no_grad():
        predicted = model(noisy_device)
    
    # Move back to CPU for plotting
    predicted = predicted.cpu()
    
    # Convert back to original scale
    max_distance = dataset.max_distance
    noisy_original = noisy[0].squeeze().numpy() * max_distance
    clean_original = clean[0].squeeze().numpy() * max_distance
    predicted_original = predicted[0].squeeze().numpy() * max_distance
    
    # Plot results
    angles = np.linspace(0, 2*np.pi, dataset.ray_count, endpoint=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(angles, clean_original, 'b-', label='True Distance', alpha=0.8)
    plt.plot(angles, noisy_original, 'r-', label='Noisy Input', alpha=0.6)
    plt.plot(angles, predicted_original, 'g-', label='Denoised Output', alpha=0.8)
    
    plt.xlabel('Angle (radians)')
    plt.ylabel('Distance')
    plt.title(f'Lidar Denoising Results (Noise Level: {dataset.noise_level:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def export_model_to_torchscript(model, input_size=60):
    """Export the trained model to TorchScript for C++ deployment"""
    model.eval()
    
    # Move model to CPU for export (C++ LibTorch typically uses CPU)
    model = model.cpu()
    print(f"Model moved to CPU for export")
    
    # Create a dummy input with the correct shape on CPU
    dummy_input = torch.randn(1, 1, input_size)
    print(f"Input device: {dummy_input.device}")
    
    # Export to TorchScript
    traced_script_module = torch.jit.trace(model, dummy_input)
    
    # Save the model
    traced_script_module.save("models/lidar_denoiser_traced.pt")
    print("Model exported to models/lidar_denoiser_traced.pt")
    
    return traced_script_module


if __name__ == "__main__":
    # First, verify the dataset
    
    try:
        dataset = LidarDataset('data/lidar_training_data.bin')
        print("Dataset verification complete!")
        
        # Train the model
        model, test_loader, full_dataset = train_model()
        
        # Test and show results
        if model is not None:
            test_model(model, test_loader, full_dataset)
            
            # Export to TorchScript
            try:
                # Ensure model is on CPU for export
                model_cpu = model.cpu()
                export_model_to_torchscript(model_cpu, full_dataset.ray_count)
                print("Model successfully exported for C++ deployment!")
            except Exception as e:
                print(f"Failed to export model: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()