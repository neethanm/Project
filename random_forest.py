import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('lulc_dataset_generated.csv')

# Assuming numerical class labels in the 'Class' column
X = df[['Red', 'Green', 'Blue', 'NIR', 'SWIR1']]
y = df['Label']

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
             xticklabels=['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land'],
             yticklabels=['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = clf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Step 5: Load Landsat bands
def load_landsat_bands(tif_file):
    """Reads the Landsat image and extracts the required bands (Red, Green, Blue, NIR, SWIR1)."""
    with rasterio.open(tif_file) as src:
        red = src.read(4)  # Band 4: Red
        green = src.read(3)  # Band 3: Green
        blue = src.read(2)  # Band 2: Blue
        nir = src.read(5)  # Band 5: NIR
        swir1 = src.read(6)  # Band 6: SWIR1
        transform = src.transform
        crs = src.crs
    return np.stack([red, green, blue, nir, swir1], axis=-1), transform, crs

# Step 6: Normalize Landsat features
def preprocess_landsat_image(landsat_bands):
    """Preprocesses the Landsat image by normalizing and flattening it."""
    # Normalize the bands to the same scale as training features
    normalized_bands = (landsat_bands - X.mean().values) / X.std().values
    return normalized_bands.reshape(-1, normalized_bands.shape[2])

# Step 7: Load and preprocess Landsat image
tif_file = r"C:\Users\91984\Downloads\LandsatImageExport.tif"  # Replace with your actual file path
landsat_bands, transform, crs = load_landsat_bands(tif_file)

print("Shape of Landsat Bands:", landsat_bands.shape)

X_landsat = preprocess_landsat_image(landsat_bands)

# Ensure consistency in feature shapes
print("Shape of Features for Prediction:", X_landsat.shape)

# Step 8: Predict LULC classes
y_landsat_pred = clf.predict(X_landsat)

# Step 9: Reshape predictions to original image dimensions
predicted_image = y_landsat_pred.reshape(landsat_bands.shape[0], landsat_bands.shape[1])

# Step 10: Save the classified result as a new TIF file
def save_classified_image(output_file, predicted_image, transform, crs):
    """Saves the classified result as a TIF file."""
    with rasterio.open(output_file, 'w', driver='GTiff', height=predicted_image.shape[0],
                        width=predicted_image.shape[1], count=1, dtype=predicted_image.dtype,
                        crs=crs, transform=transform) as dst:
        dst.write(predicted_image, 1)

output_file = 'classified_lulc.tif'
save_classified_image(output_file, predicted_image, transform, crs)

# Step 11: Display classified map
# Step 11: Display classified map
classes = ['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land']  # Replace with your actual class names

# Display the classified image using 'Spectral' colormap
plt.imshow(predicted_image, cmap='Spectral')

# Create a colorbar with numeric ticks, corresponding to the class indices
cbar = plt.colorbar()
cbar.set_ticks(np.arange(len(classes)))  # Use the number of classes as ticks
cbar.set_ticklabels(classes)  # Set the tick labels to the class names

plt.title('Classified Land Use Land Cover')
plt.show()
