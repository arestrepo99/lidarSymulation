# Define the Temporal LSTM Model
from sympy import sequence
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import time
from typing import List, Tuple
from read_lidar_temporal_data import read_lidar_temporal_data

# Set device for Apple Silicon GPU (MPS) if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


# More compact version
class HybridConvLSTMLidar(nn.Module):
    def __init__(self, input_size=120, hidden_size=16, sequence_length=10, bottleneck_size=32):
        super(HybridConvLSTMLidar, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size

        S = 2

        S1 = 2 ** (S) # 4
        S2 = 2 ** (S+1) # 8
        S3 = 2 ** (S+2) # 16
        S4 = 2 ** (S+3) # 32

        ks = 15  # Kernel size for Conv1d layers
        ps = 7   # Padding size for Conv1d layers

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, S4, ks, padding=ps, padding_mode='circular'),        #  (batch_size, 1, input_size) -> (batch_size, S4, input_size)
            nn.ReLU(),
            nn.Conv1d(S4, S3, ks, padding=ps, padding_mode='circular'),       #  (batch_size, S4, input_size) -> (batch_size, S3, input_size)
            nn.ReLU(),
            nn.MaxPool1d(2),                                                #  (batch_size, S3, input_size) -> (batch_size, S3, input_size/2)
            nn.Conv1d(S3, S2, ks, padding=ps, padding_mode='circular'),       #  (batch_size, S3, input_size/2) -> (batch_size, S2, input_size/2)
            nn.ReLU(),
            nn.Conv1d(S2, S1, ks, padding=ps, padding_mode='circular'),        #  (batch_size, S2, input_size/2) -> (batch_size, S1, input_size/2)
            nn.ReLU(),
            nn.MaxPool1d(2),                                                #  (batch_size, S1, input_size/2) -> (batch_size, S1, input_size//4)
            nn.Flatten(),                                                   #  (batch_size, S1, (input_size // 4)) -> (batch_size, S1 * (input_size // 4))
            nn.Linear(S1 * (input_size // 4), bottleneck_size),              #  (batch_size, S1 * (input_size // 4)) -> (batch_size, bottleneck_size)
            nn.ReLU()
        )
        
        # Temporal Processing of features with lstm
        # LSTM with projection - the key improvement!
        self.lstm = nn.LSTM(
            input_size=bottleneck_size,
            hidden_size=hidden_size,     
            batch_first=True,
        )
        
        # Decoder same as Encoder but backwards
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size + hidden_size, S1 * (input_size // 4)),                      # (batch_size, bottleneck_size + hidden_size) -> (batch_size, S1 * (input_size // 4))
            nn.ReLU(),
            nn.Unflatten(1, (S1, input_size // 4)),                                          # (batch_size, S1 * (input_size // 4)) -> (batch_size, S1, (input_size // 4))
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),                 # (batch_size, S1, input_size//4) -> (batch_size, S1, input_size//2)
            nn.Conv1d(S1, S2, ks, padding=ps, padding_mode='circular'),                        # (batch_size, S1, input_size//2) -> (batch_size, S2, input_size//2)
            nn.ReLU(),
            nn.Conv1d(S2, S3, ks, padding=ps, padding_mode='circular'),                       # (batch_size, S2, input_size//2) -> (batch_size, S3, input_size//2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),                 # (batch_size, S3, input_size//2) -> (batch_size, S3, input_size)
            nn.Conv1d(S3, S4, ks, padding=ps, padding_mode='circular'),                       # (batch_size, S3, input_size) -> (batch_size, S4, input_size)
            nn.ReLU(),
            nn.Conv1d(S4, 1, ks, padding=ps, padding_mode='circular'),                        # (batch_size, S4, input_size) -> (batch_size, 1, input_size)
            nn.Sigmoid()
        )
        

    def forward(self, x): # (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        # Encode all frames to bottleneck
        x = x.view(batch_size * seq_len, 1, self.input_size)
        
        encoded = self.encoder(x) # (batch_size * seq_len, bottleneck_size)


        encoded = encoded.view(batch_size, seq_len, self.bottleneck_size) # (batch_size, seq_len, bottleneck_size)
        # Process sequence with LSTM
        context, _ = self.lstm(encoded)
        # output: (batch_size, seq_len, hidden_size)
        # h_n: (batch_size, 1, hidden_size)
        # c_n: (batch_size, 1, hidden_size)

        fused = torch.cat([encoded, context], dim=2)  # (batch_size, seq_len, bottleneck_size + hidden_size)
        # last_frame = fused[:, -1, :]  # (batch_size, bottleneck_size + hidden_size)
        
        fused = fused.view(batch_size * seq_len, self.bottleneck_size + self.hidden_size) # (batch_size * seq_len, bottleneck_size + hidden_size)
        decoded = self.decoder(fused)  # (batch_size, 1, input_size) -> (batch_size, input_size)

        return decoded

# Dataset class for temporal data
class TemporalLidarDataset(Dataset):
    def __init__(self, filename, sequence_length=10):
        self.sequence_length = sequence_length
        
        # Read the temporal data using the function from the visualization code
        angles, temporal_data, noise_level, num_temporal_examples, steps_per_trajectory = read_lidar_temporal_data(filename)
        
        self.angles = angles
        self.noise_level = noise_level
        self.num_temporal_examples = num_temporal_examples
        self.steps_per_trajectory = steps_per_trajectory
        self.ray_count = len(angles)
        
        # Normalization parameter
        self.max_distance = 1200.0
        
        # Prepare sequences
        self.sequences = []
        
        for sequence in temporal_data:
            # Convert to normalized tensors
            measured_normalized = [
                torch.FloatTensor(sample['measured_dists'] / self.max_distance) 
                for sample in sequence
            ]
            true_normalized = [
                torch.FloatTensor(sample['true_dists'] / self.max_distance) 
                for sample in sequence
            ]
            
            # Create input-output pairs with the specified sequence length
            for i in range(len(sequence) - self.sequence_length):
                input_seq = torch.stack(measured_normalized[i:i+self.sequence_length])
                # output = true_normalized[i+self.sequence_length]  # Predict the denoised distance of the last step
                # self.sequences.append((input_seq, output))
                output_seq = torch.stack(true_normalized[i:i+self.sequence_length]) # Predict the denoised distances of the entire sequence
                self.sequences.append((input_seq, output_seq))

        print(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, output_seq = self.sequences[idx]
        return input_seq, output_seq

# Training function for the LSTM model
def train_temporal_model():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 200
    sequence_length = 10  # Number of previous steps to use for prediction
    patience = 5  # Early stopping patience
    
    full_dataset = TemporalLidarDataset('data/lidar_temporal_data.bin', sequence_length=sequence_length)
    
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
    model = HybridConvLSTMLidar(
        input_size=full_dataset.ray_count, 
        sequence_length=sequence_length
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    print(f"Starting training with {len(train_dataset)} sequences, {full_dataset.ray_count} rays per sample")
    
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
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # Move data to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            # loss = criterion(outputs, targets)
            loss = criterion(outputs.squeeze(1), targets.view(-1, full_dataset.ray_count))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/lidar_lstm_best.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.1f}s, '
                f'Train Loss: {avg_train_loss:.7f}, Val Loss: {avg_val_loss:.7f}, '
                f'Patience: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    # Load best model
    model.load_state_dict(torch.load('models/lidar_lstm_best.pth'))
    
    # Final test evaluation
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.6f}')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/lstm_training_history.png')
    plt.show()
    
    # Save final model
    torch.save(model.state_dict(), 'models/lidar_lstm_final.pth')
    print("Model saved as models/lidar_lstm_final.pth")
    
    return model, test_loader, full_dataset
    



# Test the LSTM model
def test_temporal_model(model, test_loader, dataset):
    if model is None:
        return
        
    model.eval()
    
    # Get a test sequence from the test loader
    test_iter = iter(test_loader)
    sequences, targets = next(test_iter)
    
    # Move to device for prediction
    sequences_device = sequences.to(device)
    
    with torch.no_grad():
        predicted = model(sequences_device)
    
    # Move back to CPU for plotting
    predicted = predicted.cpu()
    
    # Convert back to original scale
    max_distance = dataset.max_distance
    sequences_original = sequences[0].squeeze().numpy() * max_distance
    target_original = targets[0].squeeze().numpy() * max_distance
    predicted_original = predicted[0].squeeze().numpy() * max_distance
    
    # Plot results
    angles = np.linspace(0, 2*np.pi, dataset.ray_count, endpoint=False)
    
    plt.figure(figsize=(15, 10))
    
    # Plot the input sequence
    for i in range(sequences.shape[1]):
        plt.subplot(3, 4, i+1)
        plt.plot(angles, sequences_original[i], 'r-', alpha=0.7, label=f'Input Step {i}')
        plt.xlabel('Angle (radians)')
        plt.ylabel('Distance')
        plt.title(f'Input Step {i}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max_distance)
    
    # Plot the prediction vs target
    plt.subplot(3, 4, 11)
    plt.plot(angles, target_original, 'b-', label='True Next Step', alpha=0.8)
    plt.plot(angles, predicted_original, 'g-', label='Predicted Next Step', alpha=0.8)
    plt.xlabel('Angle (radians)')
    plt.ylabel('Distance')
    plt.title('Prediction vs True Next Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max_distance)
    
    # Plot the error
    plt.subplot(3, 4, 12)
    error = np.abs(predicted_original - target_original)
    plt.plot(angles, error, 'k-', label='Absolute Error')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Error')
    plt.title(f'Prediction Error (MAE: {np.mean(error):.2f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/lstm_prediction_results.png')
    plt.show()

# Function to predict next step in real-time (for deployment)
def predict_next_step(model, previous_sequence, device='cpu'):
    """
    Predict the next lidar measurement given a sequence of previous measurements
    
    Args:
        model: Trained LSTM model
        previous_sequence: Tensor of shape (sequence_length, ray_count)
        device: Device to run the prediction on
    
    Returns:
        Predicted next measurement as a tensor of shape (1, ray_count)
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        input_seq = previous_sequence.unsqueeze(0).to(device)
        prediction = model(input_seq)
        return prediction.squeeze(0).cpu()  # Remove batch dimension and return to CPU
    


def export_lstm_to_torchscript(model, sequence_length=10, ray_count=60):
    """Export the trained LSTM model to TorchScript for C++ deployment"""
    model.eval()
    
    # Move model to CPU for export
    model = model.cpu()
    print(f"Model moved to CPU for export")
    
    # Create a dummy input with the correct shape on CPU
    # Shape: (batch_size, sequence_length, input_size)
    dummy_input = torch.randn(1, sequence_length, ray_count)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input device: {dummy_input.device}")
    
    # Export to TorchScript
    traced_script_module = torch.jit.trace(model, dummy_input)
    
    # Save the model
    traced_script_module.save("models/lidar_lstm_traced.pt")
    print("LSTM model exported to models/lidar_lstm_traced.pt")
    
    return traced_script_module


def main():
    # Train the LSTM model
    model, test_loader, full_dataset = train_temporal_model()
    
    # Test and show results
    if model is not None:
        test_temporal_model(model, test_loader, full_dataset)
        
        # Example of how to use the model for prediction
        # Get a sample sequence
        test_iter = iter(test_loader)
        sequences, targets = next(test_iter)
        sample_sequence = sequences[0]  # Shape: (sequence_length, ray_count)
        
        # Predict the next step
        prediction = predict_next_step(model, sample_sequence, device)
        print(f"Prediction shape: {prediction.shape}")

    export_lstm_to_torchscript(model, sequence_length=10, ray_count=full_dataset.ray_count)

def load_best_model_and_dataset():
    # Hyperparameters
    sequence_length = 10  # Number of previous steps to use for prediction
    full_dataset = TemporalLidarDataset('data/lidar_temporal_data.bin', sequence_length=sequence_length)
    
    # Initialize model and move to device
    model = HybridConvLSTMLidar(
        input_size=full_dataset.ray_count, 
        sequence_length=sequence_length
    )

    model.load_state_dict(torch.load('models/lidar_lstm_best.pth', map_location=device))

    model.to(device)
    return full_dataset, model


def test_best_model():
    full_dataset, model = load_best_model_and_dataset()
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    if model is not None:
        test_temporal_model(model, test_loader, full_dataset)

if __name__ == "__main__":
    # test_best_model()
    main()