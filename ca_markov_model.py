import rasterio
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def read_lulc_map(lulc_map_path):
    with rasterio.open(lulc_map_path) as src:
        return src.read(1)

def compute_transition_matrix(lulc_map1, lulc_map2):
    # Flatten maps to 1D
    lulc_map1_flat = lulc_map1.flatten()
    lulc_map2_flat = lulc_map2.flatten()

    # Unique LULC classes
    all_classes = np.union1d(np.unique(lulc_map1_flat), np.unique(lulc_map2_flat))

    # Create transition matrix
    transition_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}

    # Populate transition matrix
    for from_class, to_class in zip(lulc_map1_flat, lulc_map2_flat):
        from_idx = class_to_index[from_class]
        to_idx = class_to_index[to_class]
        transition_matrix[from_idx, to_idx] += 1

    # Normalize to get probabilities
    transition_prob_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    # Convert to DataFrame
    transition_df = pd.DataFrame(
        transition_prob_matrix,
        index=[f"From {cls}" for cls in all_classes],
        columns=[f"To {cls}" for cls in all_classes]
    )
    
    return transition_df, all_classes

lulc_map1 = read_lulc_map(r"random_lulcs_for_testing\LULC_2018.tif")
lulc_map2 = read_lulc_map(r"random_lulcs_for_testing\LULC_2021.tif")
transition_df, all_classes = compute_transition_matrix(lulc_map1, lulc_map2)


def visualize_lulc_map(lulc_map, all_classes, title="LULC Map"):
    class_colors = plt.cm.get_cmap('tab20', len(all_classes))
    class_to_color = {cls: class_colors(i) for i, cls in enumerate(all_classes)}

    lulc_colored = np.array([[class_to_color[cls] for cls in row] for row in lulc_map])

    plt.imshow(lulc_colored)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_neighborhood_influence(lulc_map, neighborhood_kernel):
    """
    Apply a neighborhood influence using a kernel. The kernel size determines
    the local neighborhood considered for each pixel.
    """
    neighborhood_map = convolve(lulc_map, neighborhood_kernel, mode='nearest')
    return neighborhood_map

def simulate_lulc(lulc_map, transition_matrix, neighborhood_kernel, n_steps=5):
    """
    Simulate future LULC maps using transition probabilities and neighborhood influence.
    """
    for step in range(n_steps):
        # Apply neighborhood influence
        neighborhood_map = apply_neighborhood_influence(lulc_map, neighborhood_kernel)

        # Get transition probabilities
        transition_probs = transition_matrix[lulc_map.flatten()]

        # Update the LULC map based on the transition probabilities and neighborhood influence
        # (Here, you can apply a probabilistic model to decide state transitions)
        # This is a simple demonstration of how you can incorporate spatial influence:
        
        new_lulc_map = lulc_map.copy()
        
        for i in range(lulc_map.shape[0]):
            for j in range(lulc_map.shape[1]):
                current_class = lulc_map[i, j]
                prob_vector = transition_probs[current_class]
                
                # Modify the transition probabilities based on the neighborhood map
                # (e.g., higher probability of change to urban if surrounded by urban)
                # You can apply a function to scale or modify the probabilities here
                
                # Example: Pick a new class based on transition probabilities
                new_class = np.random.choice(all_classes, p=prob_vector)
                new_lulc_map[i, j] = new_class
        
        # Update the LULC map for the next iteration
        lulc_map = new_lulc_map
    
    return lulc_map

def simulate_ca_markov(lulc_map_paths, neighborhood_kernel, n_steps=5):

    lulc_map = read_lulc_map(lulc_map_paths[0])
    
    for t in range(1, len(lulc_map_paths)):
        next_lulc_map = read_lulc_map(lulc_map_paths[t])

        transition_df, all_classes = compute_transition_matrix(lulc_map, next_lulc_map)
        print(transition_df)
        lulc_map = simulate_lulc(lulc_map, transition_df.values, neighborhood_kernel, n_steps)

        print(f"Simulated LULC Map at time step {t}:")
        print(lulc_map)
        visualize_lulc_map(lulc_map, all_classes, title=f"Simulated LULC Map at timr step {t}")

lulc_map_paths = [r"random_lulcs_for_testing\LULC_2018.tif", r"random_lulcs_for_testing\LULC_2021.tif", r"random_lulcs_for_testing\LULC_2024.tif"]
neighborhood_kernel = np.ones((3, 3))
simulate_ca_markov(lulc_map_paths, neighborhood_kernel)
