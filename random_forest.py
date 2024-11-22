import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import numpy as np
import json

from shapely.geometry import shape, Point
from matplotlib.colors import ListedColormap

# Class mapping for land use types
class_mapping = {
    "agriculture": 1,
    "barrenland": 2,
    "buildup": 3,
    "vegetation": 4,
    "water": 5
}

# Inverse mapping for visualization
inverse_class_mapping = {v: k for k, v in class_mapping.items()}

def preprocess_raster(raster):
    """
    Replace NaN values in the raster data with the mean value of the respective band.
    """
    n_bands, n_rows, n_cols = raster.shape
    for band_idx in range(n_bands):
        band = raster[band_idx, :, :]
        if np.isnan(band).any():
            # Replace NaN values with the mean of valid pixels
            nan_mask = np.isnan(band)
            mean_value = np.nanmean(band)  # Mean of non-NaN pixels
            band[nan_mask] = mean_value
            raster[band_idx, :, :] = band
    return raster

# Step 1: Parse Geometry
def parse_geometry(geometry_str):
    geom = json.loads(geometry_str)
    if geom['type'] == 'Point':
        return geom['coordinates']  # [lon, lat]
    elif geom['type'] == 'Polygon':
        # Calculate centroid of the polygon
        poly = shape(geom)
        return [poly.centroid.x, poly.centroid.y]
    else:
        raise ValueError("Unsupported geometry type")

# Step 2: Extract Training Data
def extract_training_data(raster, csv_path, meta, class_mapping):
    df = pd.read_csv(csv_path)
    
    features = []
    target = []
    transform = meta['transform']
    
    for _, row in df.iterrows():
        geom = parse_geometry(row['.geo'])
        lon, lat = geom
        class_label = row['class']
        
        # Convert lon/lat to raster indices
        row_col = rasterio.transform.rowcol(transform, lon, lat)
        row_idx, col_idx = row_col
        
        try:
            pixel_values = raster[:, row_idx, col_idx]
            features.append(pixel_values)
            target.append(class_mapping[class_label])  # Encode label
        except IndexError:
            print(f"Skipping point ({lon}, {lat}) - outside raster bounds")
    
    features = np.array(features)
    target = np.array(target)
    return features, target

def train_random_forest(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    print("Model training completed.")
    print("Evaluation:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

def classify_raster(raster, model):
    n_bands, n_rows, n_cols = raster.shape
    reshaped = raster.reshape(n_bands, -1).T  # Reshape for model input

    nan_mask = np.isnan(reshaped).any(axis=1)
    
    predictions = np.full(reshaped.shape[0], -1, dtype=int)  # Initialize with "No Data" label
    predictions[~nan_mask] = model.predict(reshaped[~nan_mask])  # Classify valid pixels
    
    classified = predictions.reshape(n_rows, n_cols)
    return classified

def save_classified_raster(output_path, classified, meta):
    meta.update(dtype='int16', count=1, nodata=-1)  # Specify nodata value (-1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(classified.astype(np.int16), 1)
    print(f"Classified raster saved to {output_path}")

# Visualize with decoded class labels
def visualize_lulc(classified, inverse_class_mapping):
    # Define custom colors for each class, including "No Data"
    class_colors = {
        1: '#228B22',  # Agriculture - Green
        2: '#DAA520',  # Barrenland - Goldenrod
        3: '#8B0000',  # Buildup - Dark Red
        4: '#32CD32',  # Vegetation - Lime Green
        5: '#4682B4',  # Water - Steel Blue
        -1: '#FFFFFF'  # No Data - White
    }

    # Create colormap and corresponding labels
    colors = [class_colors.get(cls, '#000000') for cls in sorted(class_colors.keys())]
    cmap = ListedColormap(colors)

    # Get unique class values from the classified raster
    unique_classes = np.unique(classified)

    # Ensure legend matches the unique classes
    legend_labels = [inverse_class_mapping.get(cls, "No Data") for cls in unique_classes]
    patches = [
        plt.matplotlib.patches.Patch(color=class_colors.get(cls, '#000000'), label=label)
        for cls, label in zip(unique_classes, legend_labels)
    ]

    # Plot the raster with the custom colormap
    plt.figure(figsize=(10, 8))
    plt.imshow(classified, cmap=cmap, interpolation='nearest')

    # Add legend and title
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title("LULC Map")
    plt.axis('off')
    st.pyplot(plt)  # Use Streamlit to display the plot

def display_lulc(landsat_path):
    csv_path = "little_more_built_up.csv"
    output_path = "LULC_results/lulc_map.tif"
    
    with rasterio.open(landsat_path) as src:
        raster = src.read()
        meta = src.meta

    st.write("Preprocessing raster to handle NaN values...")

    st.write("Extracting training data...")
    features, target = extract_training_data(raster, csv_path, meta, class_mapping)

    st.write("Training Random Forest model...")
    rf_model = train_random_forest(features, target)

    st.write("Classifying raster...")
    classified = classify_raster(raster, rf_model)

    save_classified_raster(output_path, classified, meta)
    visualize_lulc(classified, inverse_class_mapping)

st.title("Land Use Land Cover (LULC) Classification")
landsat_file = st.file_uploader("Upload Landsat Image", type=["tif"])

if landsat_file:
    display_lulc(landsat_file)
