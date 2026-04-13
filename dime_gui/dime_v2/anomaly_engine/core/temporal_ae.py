import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=256, seq_length=16):
        super().__init__()
        self.seq_length = seq_length
        
        # Encoder: Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder: Future frame predictor
        self.decoder = nn.LSTM(
            input_size=latent_dim*2,  # Bidirectional
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Reconstruction head
        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, input_dim)
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, input_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # Encode sequence
        enc_out, _ = self.encoder(x)
        
        # Decode for reconstruction
        recon_out, _ = self.decoder(enc_out)
        reconstructed = self.recon_head(recon_out)
        
        # Predict future frames (shifted by one)
        pred_out, _ = self.decoder(enc_out[:, :-1, :])
        predictions = self.pred_head(pred_out)
        
        return reconstructed, predictions

class MotionAnomalyScorer:
    def __init__(self, model, device, seq_length=16, threshold=0.2):
        self.model = model.to(device)
        self.device = device
        self.seq_length = seq_length
        self.threshold = threshold
        self.feature_buffer = []
        
    def update(self, features):
        """Update with new features and return anomaly score"""
        # Detach and move to CPU to save GPU memory
        features = features.detach().cpu().numpy()
        
        # Add to buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.seq_length:
            self.feature_buffer.pop(0)
            
        if len(self.feature_buffer) < self.seq_length:
            return 0.0  # Not enough frames
            
        # Create sequence tensor
        seq_tensor = torch.tensor(np.array(self.feature_buffer), dtype=torch.float32)
        seq_tensor = seq_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            reconstructed, predictions = self.model(seq_tensor)
            
            # Reconstruction error (current sequence)
            rec_error = F.mse_loss(reconstructed, seq_tensor).item()
            
            # Prediction error (next frame)
            actual_next = seq_tensor[:, 1:, :]
            pred_error = F.mse_loss(predictions, actual_next).item()
            
        # Combine errors
        anomaly_score = (rec_error + pred_error) / 2
        return max(0, anomaly_score - self.threshold)