import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def read_lulc_map(lulc_map_path):
    with rasterio.open(lulc_map_path) as src:
        return src.read(1)

def compute_transition_matrix(lulc_map1, lulc_map2):
    lulc_map1_flat = lulc_map1.flatten()
    lulc_map2_flat = lulc_map2.flatten()

    all_classes = np.union1d(np.unique(lulc_map1_flat), np.unique(lulc_map2_flat))

    transition_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}

    for from_class, to_class in zip(lulc_map1_flat, lulc_map2_flat):
        from_idx = class_to_index[from_class]
        to_idx = class_to_index[to_class]
        transition_matrix[from_idx, to_idx] += 1

    transition_prob_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    return transition_prob_matrix, all_classes

def visualize_lulc_map(lulc_map, all_classes, title="LULC Map", colormap='Spectral'):
    class_colors = plt.cm.get_cmap(colormap, len(all_classes))
    class_to_color = {cls: class_colors(i) for i, cls in enumerate(all_classes)}

    lulc_colored = np.array([[class_to_color[cls] for cls in row] for row in lulc_map])

    plt.imshow(lulc_colored)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_neighborhood_influence(lulc_map, neighborhood_kernel):
    neighborhood_map = convolve(lulc_map, neighborhood_kernel, mode='nearest')
    return neighborhood_map

def simulate_lulc(lulc_map, transition_matrix, neighborhood_kernel, all_classes, n_steps=5):
    for step in range(n_steps):
        neighborhood_map = apply_neighborhood_influence(lulc_map, neighborhood_kernel)
        transition_probs = transition_matrix[lulc_map.flatten()]
        new_lulc_map = lulc_map.copy()

        for i in range(lulc_map.shape[0]):
            for j in range(lulc_map.shape[1]):
                current_class = lulc_map[i, j]
                prob_vector = transition_probs[current_class]
                neighborhood_influence = neighborhood_map[i, j]

                prob_vector = prob_vector * neighborhood_influence
                prob_sum = np.sum(prob_vector)
                if prob_sum > 0:
                    prob_vector = prob_vector / prob_sum
                else:
                    prob_vector = np.ones(len(all_classes)) / len(all_classes)

                new_class = np.random.choice(all_classes, p=prob_vector)
                new_lulc_map[i, j] = new_class

        lulc_map = new_lulc_map

    return lulc_map

def predict_lulc_for_2027(lulc_map_paths, neighborhood_kernel, n_steps=5):
    lulc_map = read_lulc_map(lulc_map_paths[0])
    transition_matrix_accum = None
    all_classes = None

    for t in range(1, len(lulc_map_paths)):
        next_lulc_map = read_lulc_map(lulc_map_paths[t])
        transition_prob_matrix, current_classes = compute_transition_matrix(lulc_map, next_lulc_map)

        if transition_matrix_accum is None:
            transition_matrix_accum = transition_prob_matrix
            all_classes = current_classes
        else:
            transition_matrix_accum += transition_prob_matrix

        lulc_map = next_lulc_map

    transition_matrix_accum = transition_matrix_accum / (len(lulc_map_paths) - 1)

    predicted_lulc_map = simulate_lulc(lulc_map, transition_matrix_accum, neighborhood_kernel, all_classes, n_steps)

    visualize_lulc_map(predicted_lulc_map, all_classes, title="Predicted LULC Map for 2027")

lulc_map_paths = [r"random_lulcs_for_testing/LULC_2018.tif", r"random_lulcs_for_testing/LULC_2021.tif", r"random_lulcs_for_testing/LULC_2024.tif"]
neighborhood_kernel = np.ones((3, 3))
predict_lulc_for_2027(lulc_map_paths, neighborhood_kernel)
