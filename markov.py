import numpy as np
import rasterio
import random
from PIL import Image

def extract_colors(img):
    """Extract unique pixel values and their corresponding RGB colors."""
    unique_classes = np.unique(img)
    color_mapping = {}
    for cls in unique_classes:
        color_mapping[cls] = [cls, cls, cls]  # If single-band, treat as grayscale
    return color_mapping

def generate_transition_matrix(img1, img2):
    """Generate the transition matrix between two LULC maps."""
    unique_classes = np.unique(img1)
    num_classes = len(unique_classes)
    class_indices = {cls: idx for idx, cls in enumerate(unique_classes)}
    transition_matrix = np.zeros((num_classes, num_classes), dtype=float)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            class1 = img1[i, j]
            class2 = img2[i, j]
            if class1 in class_indices and class2 in class_indices:
                transition_matrix[class_indices[class1], class_indices[class2]] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    np.seterr(divide='ignore', invalid='ignore')  # Suppress warnings for zero rows
    transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums != 0)

    return transition_matrix, class_indices

def apply_constraints(predicted_image, road_network, dem, slope, color_mapping):
    """Modify the predicted LULC using road networks, DEM, and slope constraints."""
    for i in range(predicted_image.shape[0]):
        for j in range(predicted_image.shape[1]):
            # Ensure that the indices are within bounds of the provided arrays
            if i < road_network.shape[0] and j < road_network.shape[1]:
                if road_network[i, j] > 0:  # Close to road
                    predicted_image[i, j] = color_mapping[min(color_mapping.keys())]  # Prefer smallest class (Urban)
                elif slope[i, j] > 15:  # High slope areas
                    predicted_image[i, j] = color_mapping[max(color_mapping.keys())]  # Prefer largest class (Vegetation)
                elif dem[i, j] < 10:  # Low-lying areas
                    predicted_image[i, j] = color_mapping[sorted(color_mapping.keys())[1]]  # Prefer second smallest (Water)
    return predicted_image

def markovPredict(img1, img2, road_network, dem, slope, output_path, transform, crs):
    """Predict LULC using Markov model and apply spatial constraints."""
    color_mapping = extract_colors(img1)  # Extract colors dynamically
    transition_matrix, class_indices = generate_transition_matrix(img1, img2)

    predicted_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i, j] in class_indices:
                x = random.random()
                y = img2[i, j]
                cumulative_prob = 0
                for k, cls in enumerate(class_indices):
                    cumulative_prob += transition_matrix[class_indices[y], k]
                    if x <= cumulative_prob:
                        predicted_image[i, j] = color_mapping[cls]
                        break

    # Apply road network, DEM, and slope constraints
    predicted_image = apply_constraints(predicted_image, road_network, dem, slope, color_mapping)

    # Save the predicted image
    with rasterio.open(output_path, 'w', driver='GTiff', height=img1.shape[0],
                       width=img1.shape[1], count=3, dtype='uint8', crs=crs, transform=transform) as dst:
        for band in range(3):
            dst.write(predicted_image[..., band], band + 1)

    # Save as PNG for visualization
    Image.fromarray(predicted_image, 'RGB').save(output_path.replace(".tif", ".png"))
    print(f"Prediction saved to {output_path} and visualization saved as PNG.")

# Load LULC, road network, DEM, and slope data
with rasterio.open(r"random_lulcs_for_testing/LULC_2018.tif") as src:
    img1 = src.read(1)
    transform = src.transform
    crs = src.crs

with rasterio.open(r"random_lulcs_for_testing/LULC_2024.tif") as src:
    img2 = src.read(1)

with rasterio.open(r"osm_results\road_network.tif") as src:
    road_network = src.read(1)

with rasterio.open(r"random_lulcs_for_testing/clipped_dem.tif") as src:
    dem = src.read(1)

with rasterio.open(r"random_lulcs_for_testing/slope_dem.tif") as src:
    slope = src.read(1)

# Ensure all arrays have the same size
if img1.shape != road_network.shape:
    road_network = road_network[:img1.shape[0], :img1.shape[1]]
if img1.shape != dem.shape:
    dem = dem[:img1.shape[0], :img1.shape[1]]
if img1.shape != slope.shape:
    slope = slope[:img1.shape[0], :img1.shape[1]]

# Predict and save the output
output_path = r"predicted_output.tif"
markovPredict(img1, img2, road_network, dem, slope, output_path, transform, crs)
