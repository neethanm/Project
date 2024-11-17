import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import numpy as np

def display_lulc(path):
    df = pd.read_csv('lulc_dataset_generated.csv')
    X = df[['Red', 'Green', 'Blue', 'NIR', 'SWIR1']]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Classification Accuracy: {accuracy:.2f}")

    st.write("\nClassification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land']))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])

    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land'],
                yticklabels=['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # plt.show()
    st.pyplot(fig)

    importances = clf.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.write("\n###Feature Importances:")
    st.write(feature_importance_df)

    def load_landsat_bands(path):
        """Reads the Landsat image and extracts the required bands (Red, Green, Blue, NIR, SWIR1)."""
        with rasterio.open(path) as src:
            red = src.read(4)  # Band 4: Red
            green = src.read(3)  # Band 3: Green
            blue = src.read(2)  # Band 2: Blue
            nir = src.read(5)  # Band 5: NIR
            swir1 = src.read(6)  # Band 6: SWIR1
            transform = src.transform
            crs = src.crs
        return np.stack([red, green, blue, nir, swir1], axis=-1), transform, crs
    
    def preprocess_landsat_image(landsat_bands):
        # Normalize the bands to the same scale as training features
        normalized_bands = (landsat_bands - X.mean().values) / X.std().values
        return normalized_bands.reshape(-1, normalized_bands.shape[2])
    
    landsat_bands, transform, crs = load_landsat_bands(path)
    print("Shape of Landsat Bands:", landsat_bands.shape)
    X_landsat = preprocess_landsat_image(landsat_bands)
    print("Shape of Features for Prediction:", X_landsat.shape)
    y_landsat_pred = clf.predict(X_landsat)
    predicted_image = y_landsat_pred.reshape(landsat_bands.shape[0], landsat_bands.shape[1])

    def save_classified_image(output_file, predicted_image, transform, crs):
        with rasterio.open(output_file, 'w', driver='GTiff', height=predicted_image.shape[0],
                            width=predicted_image.shape[1], count=1, dtype=predicted_image.dtype,
                            crs=crs, transform=transform) as dst:
            dst.write(predicted_image, 1)

    output_file = 'classified_lulc.tif'
    save_classified_image(output_file, predicted_image, transform, crs)

    classes = ['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land']

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(predicted_image, cmap='Spectral')

    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_ticks(np.arange(len(classes)))
    cbar.set_ticklabels(classes) 
    plt.title('Classified Land Use Land Cover')
    st.pyplot(fig)
