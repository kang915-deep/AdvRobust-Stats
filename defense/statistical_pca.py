import torch
import numpy as np
from sklearn.decomposition import PCA

class PCADefense:
    def __init__(self, n_components=128):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, features_np):
        """Fit PCA on clean latent features."""
        self.pca.fit(features_np)
        self.is_fitted = True
        
    def reconstruct(self, features_tensor):
        """Project to PCA subspace and reconstruct back."""
        if not self.is_fitted:
            return features_tensor
            
        device = features_tensor.device
        dtype = features_tensor.dtype
        
        # Convert to numpy
        x = features_tensor.detach().cpu().numpy()
        
        # PCA projection and reconstruction
        x_pca = self.pca.transform(x)
        x_rec = self.pca.inverse_transform(x_pca)
        
        # Convert back to tensor
        return torch.tensor(x_rec, device=device, dtype=dtype)
        
    def get_explained_variance(self, features_np):
        """Useful for variance analysis evaluation."""
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet.")
        x_pca = self.pca.transform(features_np)
        return np.var(x_pca, axis=0)
