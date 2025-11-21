import numpy as np
import os
from pathlib import Path
from sklearn.decomposition import PCA


class TestPCATransformation:
    """Test that PCA model correctly transforms latent features to x,y,z coordinates"""
    
    def test_pca_transforms_latent_to_stored_coordinates(self):
        """
        Test that applying PCA to the 16-dimensional latent features
        produces the same x,y,z coordinates stored in reduced_latent_matrix.
        """
        repo_root = Path(__file__).resolve().parents[1]
        
        # Load the main dataset with reduced coordinates
        dataset_path = repo_root / "python_server" / "latent_param_dataset_16.npz"
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"
        
        dataset = np.load(dataset_path)
        print(f"Available keys in dataset: {list(dataset.keys())}")
        
        # Load the full latent matrix (16 features) for PCA training
        latent_dataset_path = repo_root / "python_server" / "benjolin_training" / "latent_param_dataset_16.npy.npz"
        assert latent_dataset_path.exists(), f"Latent dataset not found at {latent_dataset_path}"
        
        latent_dataset = np.load(latent_dataset_path)
        print(f"Available keys in latent dataset: {list(latent_dataset.keys())}")
        
        assert 'latent_matrix' in latent_dataset, "latent_matrix not found in dataset"
        latent_matrix = latent_dataset['latent_matrix']
        print(f"Latent matrix shape: {latent_matrix.shape}")
        
        # Fit PCA model on the full latent matrix (same as BenjolinTrainer does)
        pca_model = PCA(n_components=3)
        pca_model.fit(latent_matrix)
        print(f"PCA explained variance ratio: {pca_model.explained_variance_ratio_}")
        
        # Load the stored reduced coordinates from the main dataset
        reduced_latent = dataset['reduced_latent_matrix']
        print(f"Reduced latent matrix shape: {reduced_latent.shape}")
        
        # Fix PCA component signs to match stored coordinates (same as BenjolinTrainer does)
        first_pca = pca_model.transform(latent_matrix[0].reshape(1, -1))[0]
        first_stored = reduced_latent[0][:3]
        
        sign_x = 1 if first_pca[0] * first_stored[0] > 0 else -1
        sign_y = 1 if first_pca[1] * first_stored[1] > 0 else -1
        sign_z = 1 if first_pca[2] * first_stored[2] > 0 else -1
        
        if sign_x == -1:
            pca_model.components_[0] *= -1
        if sign_y == -1:
            pca_model.components_[1] *= -1
        if sign_z == -1:
            pca_model.components_[2] *= -1
        
        print(f"PCA component signs fixed: x={sign_x}, y={sign_y}, z={sign_z}")
        
        # Transform the latent matrix to get x,y,z coordinates (after sign correction)
        pca_coords = pca_model.transform(latent_matrix)
        print(f"PCA transformed coordinates shape: {pca_coords.shape}")
        
        # Test a few random samples
        num_samples_to_test = min(10, len(latent_matrix))
        test_indices = np.random.choice(len(latent_matrix), num_samples_to_test, replace=False)
        
        print(f"\nTesting {num_samples_to_test} random samples:")
        all_match = True
        
        for idx in test_indices:
            # Get the 16-dimensional latent features
            latent_features = latent_matrix[idx].reshape(1, -1)
            
            # Apply PCA transformation
            pca_result = pca_model.transform(latent_features)
            x_pca, y_pca, z_pca = pca_result[0]
            
            # Get stored coordinates
            x_stored, y_stored, z_stored = reduced_latent[idx][:3]
            
            # Calculate difference
            diff_x = abs(x_pca - x_stored)
            diff_y = abs(y_pca - y_stored)
            diff_z = abs(z_pca - z_stored)
            
            # Check if they match (with small tolerance for floating point errors)
            tolerance = 1e-5
            matches = (diff_x < tolerance and diff_y < tolerance and diff_z < tolerance)
            
            print(f"\nIndex {idx}:")
            print(f"  PCA result:    x={x_pca:.6f}, y={y_pca:.6f}, z={z_pca:.6f}")
            print(f"  Stored coords: x={x_stored:.6f}, y={y_stored:.6f}, z={z_stored:.6f}")
            print(f"  Differences:   dx={diff_x:.6e}, dy={diff_y:.6e}, dz={diff_z:.6e}")
            print(f"  Match: {matches}")
            
            if not matches:
                all_match = False
        
        # Overall test assertion
        assert all_match, (
            "PCA transformation does not match stored coordinates. "
            "This suggests the PCA model or dataset may have changed."
        )
        
        print("\n✓ All tested samples match! PCA transformation is consistent.")
    
    def test_single_point_pca_transformation(self):
        """
        Test PCA transformation for a single specific point (index 0)
        to verify the exact workflow used in get_new_coordinates.
        """
        repo_root = Path(__file__).resolve().parents[1]
        
        # Load datasets
        dataset_path = repo_root / "python_server" / "latent_param_dataset_16.npz"
        latent_dataset_path = repo_root / "python_server" / "benjolin_training" / "latent_param_dataset_16.npy.npz"
        
        dataset = np.load(dataset_path)
        latent_dataset = np.load(latent_dataset_path)
        
        latent_matrix = latent_dataset['latent_matrix']
        reduced_latent = dataset['reduced_latent_matrix']
        
        # Fit PCA
        pca_model = PCA(n_components=3)
        pca_model.fit(latent_matrix)
        
        # Fix PCA component signs to match stored coordinates (same as BenjolinTrainer does)
        first_pca = pca_model.transform(latent_matrix[0].reshape(1, -1))[0]
        first_stored = reduced_latent[0][:3]
        
        sign_x = 1 if first_pca[0] * first_stored[0] > 0 else -1
        sign_y = 1 if first_pca[1] * first_stored[1] > 0 else -1
        sign_z = 1 if first_pca[2] * first_stored[2] > 0 else -1
        
        if sign_x == -1:
            pca_model.components_[0] *= -1
        if sign_y == -1:
            pca_model.components_[1] *= -1
        if sign_z == -1:
            pca_model.components_[2] *= -1
        
        print(f"PCA component signs fixed: x={sign_x}, y={sign_y}, z={sign_z}")
        
        # Test index 0 (first point)
        test_index = 0
        latent_features = latent_matrix[test_index].reshape(1, -1)
        
        print(f"\nTesting index {test_index}:")
        print(f"Latent features shape: {latent_features.shape}")
        print(f"Latent features (first 5): {latent_features[0][:5]}")
        
        # Apply PCA (same as in get_new_coordinates)
        pca_result = pca_model.transform(latent_features)
        x_pca, y_pca, z_pca = pca_result[0]
        
        # Get expected coordinates
        x_expected, y_expected, z_expected = reduced_latent[test_index][:3]
        
        print(f"PCA result:    ({x_pca:.6f}, {y_pca:.6f}, {z_pca:.6f})")
        print(f"Expected:      ({x_expected:.6f}, {y_expected:.6f}, {z_expected:.6f})")
        
        # Assert with tolerance
        tolerance = 1e-5
        assert abs(x_pca - x_expected) < tolerance, f"X coordinate mismatch: {x_pca} vs {x_expected}"
        assert abs(y_pca - y_expected) < tolerance, f"Y coordinate mismatch: {y_pca} vs {y_expected}"
        assert abs(z_pca - z_expected) < tolerance, f"Z coordinate mismatch: {z_pca} vs {z_expected}"
        
        print("✓ Single point PCA transformation matches expected coordinates!")