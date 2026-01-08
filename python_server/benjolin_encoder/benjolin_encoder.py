import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import torchaudio
from dataloader import get_features


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation, device):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.dense1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation1 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation2 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense3 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.activation3 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.sequential = nn.Sequential(self.dense1, self.activation1, self.dense2, self.activation2,
                                        self.dense3, self.activation3)

        self.denseMu = nn.Linear(self.hidden_dim // 2, self.latent_dim)
        self.denseLogVar = nn.Linear(self.hidden_dim // 2, self.latent_dim)

    def reparameterization(self, mu, log_variance):
        sigma = 0.5 * torch.exp(log_variance)
        return mu + sigma * dists.Normal(0, 1).sample(mu.shape).to(self.device)

    def forward(self, x):
        h = self.sequential(x)
        mu = self.denseMu(h)
        log_variance = self.denseLogVar(h)

        z = self.reparameterization(mu, log_variance)
        return z, mu, log_variance

    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation, device):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.dense1 = nn.Linear(self.latent_dim, self.hidden_dim // 2)
        self.activation1 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        self.activation2 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation3 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.activation4 = nn.ReLU()

        self.sequential = nn.Sequential(self.dense1, self.activation1, self.dense2, self.activation2, 
                                        self.dense3, self.activation3, self.dense4) #, self.activation4)
    
    def forward(self, z):
        x_hat = self.sequential(z)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation='sigmoid', device='cuda'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, activation, self.device)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, activation, self.device)

    def forward(self, x):
        x = x.flatten()
        z, _, _ = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class BenjolinEncoder:
    def __init__(self, model_path='./model', latent_dim=16, input_dim=36):
        """
        Initialize BenjolinEncoder with VAE and PCA models.
        
        Args:
            model_path: Path to the saved VAE model weights
            pca_model_path: Path to the saved PCA model
            latent_dim: Dimensionality of VAE latent space (default: 16)
            input_dim: Input feature dimension (default: 36 for bag-of-frames with 18 features * 2 stats)
        """
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Calculate hidden dimension
        hidden_dim = input_dim // 2
        
        # Initialize VAE model with CPU device for macOS compatibility
        self.vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device='cpu')
        print("VAE model created")
        
        # Set default dtype
        torch.set_default_dtype(torch.float32)
        
        # Load trained VAE weights
        if os.path.exists(model_path):
            self.vae.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.vae.eval()  # Set to evaluation mode
            print(f"VAE model loaded from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")
        
        # Load dataset and fit PCA model
        dataset_path = os.path.join(os.path.dirname(model_path), 'latent_param_dataset_16.npy.npz')
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            dataset = np.load(dataset_path)
            print(f"Available keys in dataset: {list(dataset.keys())}")
            
            # Extract latent matrix from dataset
            if 'latent_matrix' in dataset:
                latent_matrix = dataset['latent_matrix']
                print(f"Fitting PCA on latent matrix of shape {latent_matrix.shape}")
                self.pca_model = PCA(n_components=3)
                self.pca_model.fit(latent_matrix)
                print("PCA model fitted successfully")
                print(f"PCA explained variance ratio: {self.pca_model.explained_variance_ratio_}")
                
                # Load the reference dataset with stored coordinates to fix PCA signs
                reference_dataset_path = os.path.join(os.path.dirname(model_path), '..', 'latent_param_dataset_16.npz')
                if os.path.exists(reference_dataset_path):
                    reference_dataset = np.load(reference_dataset_path)
                    if 'reduced_latent_matrix' in reference_dataset:
                        reduced_latent = reference_dataset['reduced_latent_matrix']
                        
                        # Transform first point with PCA
                        first_pca = self.pca_model.transform(latent_matrix[0].reshape(1, -1))[0]
                        first_stored = reduced_latent[0][:3]
                        
                        # Detect sign flips for each component
                        sign_x = 1 if first_pca[0] * first_stored[0] > 0 else -1
                        sign_y = 1 if first_pca[1] * first_stored[1] > 0 else -1
                        sign_z = 1 if first_pca[2] * first_stored[2] > 0 else -1
                        
                        # Fix PCA component signs by flipping the components in the model
                        if sign_x == -1:
                            self.pca_model.components_[0] *= -1
                        if sign_y == -1:
                            self.pca_model.components_[1] *= -1
                        if sign_z == -1:
                            self.pca_model.components_[2] *= -1
                        
                        print(f"PCA component signs corrected: x={sign_x}, y={sign_y}, z={sign_z}")
                        if sign_x == -1 or sign_y == -1 or sign_z == -1:
                            print("  (Some components were flipped to match stored coordinates)")
                    else:
                        print("Warning: reduced_latent_matrix not found in reference dataset")
                else:
                    print(f"Warning: Reference dataset not found at {reference_dataset_path}")
                
                # Transform the training data to see the expected coordinate range
                pca_coords = self.pca_model.transform(latent_matrix)
                print(f"PCA coordinate ranges from training data:")
                print(f"  x=[{pca_coords[:, 0].min():.2f}, {pca_coords[:, 0].max():.2f}]")
                print(f"  y=[{pca_coords[:, 1].min():.2f}, {pca_coords[:, 1].max():.2f}]")
                print(f"  z=[{pca_coords[:, 2].min():.2f}, {pca_coords[:, 2].max():.2f}]")
                
                # Store statistics for normalization
                self.latent_mean = np.mean(latent_matrix, axis=0)
                self.latent_std = np.std(latent_matrix, axis=0)
                print(f"Latent mean range: [{self.latent_mean.min():.2f}, {self.latent_mean.max():.2f}]")
                print(f"Latent std range: [{self.latent_std.min():.2f}, {self.latent_std.max():.2f}]")
            elif 'reduced_latent_matrix' in dataset:
                # If only reduced version exists, we'll use first 3 VAE dimensions
                print("Only reduced_latent_matrix found. Will use first 3 VAE latent dimensions.")
                reduced = dataset['reduced_latent_matrix']
                print(f"Reduced latent matrix shape: {reduced.shape}")
                print(f"Coordinate ranges: x=[{reduced[:, 0].min():.2f}, {reduced[:, 0].max():.2f}], "
                      f"y=[{reduced[:, 1].min():.2f}, {reduced[:, 1].max():.2f}], "
                      f"z=[{reduced[:, 2].min():.2f}, {reduced[:, 2].max():.2f}]")
                self.pca_model = None
                self.latent_mean = None
                self.latent_std = None
            else:
                print("Warning: Could not find latent_matrix or reduced_latent_matrix in dataset.")
                self.pca_model = None
                self.latent_mean = None
                self.latent_std = None
        else:
            print(f"Warning: Dataset file not found at {dataset_path}.")
            self.pca_model = None
            self.latent_mean = None
            self.latent_std = None 

    def get_new_coordinates(self, sound_buffer):
        """
        Encode audio buffer into 3D coordinates for visualization.
        
        Args:
            sound_buffer: Audio signal as numpy array or torch tensor
            
        Returns:
            tuple: (x, y, z) coordinates in 3D space, or None if PCA model not available
        """
        print(f"Processing audio buffer of length {len(sound_buffer)}")
        
        # Convert to torch tensor if needed
        if isinstance(sound_buffer, np.ndarray):
            sound_buffer = torch.from_numpy(sound_buffer).float()

        print(f"Audio stats: min={sound_buffer.min():.4f}, max={sound_buffer.max():.4f}, std={sound_buffer.std():.4f}")

        
        # Extract all features to match training: 13 MFCCs + spectral centroid + 4 audio features = 18 features
        # Sample rate assumption (adjust if needed)
        sample_rate = 44100
        
        # Extract MFCCs (13 coefficients)
        MFCC = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 1024, "win_length": 1024, "hop_length": 64, "pad": 0, "n_mels": 101, "center": False}
        )
        mfcc = MFCC(sound_buffer)  # Shape: (13, num_frames)
        
        # Extract spectral centroid
        get_spectral_centroid = torchaudio.transforms.SpectralCentroid(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=64,
            pad=0
        )
        sp_centroid = get_spectral_centroid(sound_buffer + 0.001).unsqueeze(0)  # Shape: (1, num_frames)
        
        # Extract 4 audio features using get_features
        rms, zcr, spectral_flux, spectral_flatness = get_features(sound_buffer, device='cpu')
        
        # Find minimum frame count across all features to ensure consistent dimensions
        min_frames = min(mfcc.shape[1], sp_centroid.shape[1], rms.shape[0], 
                         zcr.shape[0], spectral_flux.shape[0], spectral_flatness.shape[0])
        
        # Truncate all features to the minimum frame count
        mfcc = mfcc[:, :min_frames]
        sp_centroid = sp_centroid[:, :min_frames]
        rms = rms[:min_frames]
        zcr = zcr[:min_frames]
        spectral_flux = spectral_flux[:min_frames]
        spectral_flatness = spectral_flatness[:min_frames]
        
        # Stack all features: 13 MFCCs + 1 centroid + 4 features = 18 features
        features_tensor = torch.vstack([mfcc, sp_centroid, rms.unsqueeze(0), zcr.unsqueeze(0), 
                                        spectral_flux.unsqueeze(0), spectral_flatness.unsqueeze(0)])
        
        # Compute mean and std for each feature (bag-of-frames representation)
        mean = torch.mean(features_tensor, dim=1, keepdim=True)  # Shape: (18, 1)
        std = torch.std(features_tensor, dim=1, keepdim=True)    # Shape: (18, 1)
        
        # Concatenate mean and std: 18 features × 2 = 36 dimensions
        features = torch.hstack([mean, std]).flatten()  # Shape: (36,)

        print("Features (36-dim) =", features)
        print("Features hash:", hash(tuple(float(x) for x in features)))

        
        # Debug: Check for NaN in features
        if torch.isnan(features).any():
            print("WARNING: NaN detected in features before encoding!")
            print(f"Features: {features}")
            print(f"Mean: {mean.flatten()}")
            print(f"Std: {std.flatten()}")
        
        # Encode features into VAE latent space
        with torch.no_grad():
            z, mu, sigma = self.vae.encoder.forward(features)
            print("mu =", mu)
            print("mu hash:", hash(tuple(float(x) for x in mu)))
        
        # Debug: Check for NaN in encoder output
        if torch.isnan(mu).any():
            print("ERROR: NaN detected in encoder output mu!")
            print(f"mu: {mu}")
            print(f"z: {z}")
            print(f"sigma: {sigma}")
            print("Returning default coordinates (0, 0, 0)")
            return (0.0, 0.0, 0.0)
        
        # Convert to numpy
        mu_np = mu.cpu().detach().numpy()
        
        # Check for NaN in numpy array
        if np.isnan(mu_np).any():
            print("ERROR: NaN detected in mu_np after conversion to numpy!")
            print(f"mu_np: {mu_np}")
            print("Returning default coordinates (0, 0, 0)")
            return (0.0, 0.0, 0.0)
        
        # Reshape to 2D array for PCA (1 sample, 16 features)
        latent_matrix = mu_np.reshape(1, -1)
    
        print("Latent matrix shape:", latent_matrix.shape)
        print("Latent matrix:", latent_matrix)
        
        # Reduce dimensionality to 3D using PCA
        if self.pca_model is not None:
            print("Using PCA model for dimensionality reduction")
            # Use transform (not fit) since PCA was already fitted during initialization
            pca_latent = self.pca_model.transform(latent_matrix)
            x, y, z = pca_latent[0]
            print("PCA coords:", x, y, z)
            return (float(x), float(y), float(z))
        else:
            print("Warning: PCA model not available, returning first 3 latent dimensions")
            x, y, z = mu_np[0], mu_np[1], mu_np[2]
            return (float(x), float(y), float(z))

# Backwards-compatible alias for older code that may still import BenjolinTrainer
BenjolinTrainer = BenjolinEncoder