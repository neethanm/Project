import rasterio
import numpy as np
import pandas as pd

lulc_map1_path = r"random_lulcs_for_testing\LULC_2018.tif"
lulc_map2_path = r"random_lulcs_for_testing\LULC_2021.tif"

def compute_transition_matrix(lulc_map1_path, lulc_map2_path):
    try:
        # Open the two LULC maps
        with rasterio.open(lulc_map1_path) as src1, rasterio.open(lulc_map2_path) as src2:
            # Ensure the maps have the same shape
            if src1.shape != src2.shape:
                raise ValueError("LULC maps must have the same dimensions.")
            
            # Read the LULC data as numpy arrays
            lulc_map1 = src1.read(1)  # Read the first band
            lulc_map2 = src2.read(1)  # Read the first band
            
            # Flatten the arrays to 1D for easier processing
            lulc_map1_flat = lulc_map1.flatten()
            lulc_map2_flat = lulc_map2.flatten()
            
            # Get unique classes in both maps
            unique_classes1 = np.unique(lulc_map1_flat)
            unique_classes2 = np.unique(lulc_map2_flat)
            
            # Combine classes to ensure all are considered
            all_classes = np.union1d(unique_classes1, unique_classes2)
            
            # Initialize the transition matrix
            transition_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
            
            # Create a mapping from class to index for the matrix
            class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}
            
            # Populate the transition matrix
            for from_class, to_class in zip(lulc_map1_flat, lulc_map2_flat):
                from_idx = class_to_index[from_class]
                to_idx = class_to_index[to_class]
                transition_matrix[from_idx, to_idx] += 1
            
            # Normalize to get probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_prob_matrix = transition_matrix / np.where(row_sums > 0, row_sums, 1)
            
            # Convert to a DataFrame for better visualization
            transition_df = pd.DataFrame(
                transition_prob_matrix,
                index=[f"From {cls}" for cls in all_classes],
                columns=[f"To {cls}" for cls in all_classes]
            )
            
            # Print the transition matrix
            print("Transition Matrix (Probabilities):")
            print(transition_df)
            
            return transition_df
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
transition_matrix = compute_transition_matrix(lulc_map1_path, lulc_map2_path)
