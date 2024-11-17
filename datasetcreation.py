import pandas as pd
import numpy as np

# Define the classes and their labels
classes = {
    'Urban': {'label': 0, 'color': (200, 200, 200)},  # Light Gray
    'Vegetation': {'label': 1, 'color': (0, 128, 0)},  # Green
    'Water': {'label': 2, 'color': (0, 0, 255)},  # Blue
    'Agriculture': {'label': 3, 'color': (255, 255, 0)},  # Yellow
    'Open Land': {'label': 4, 'color': (210, 180, 140)}  # Tan
}

# Create synthetic band values (simulated ranges for each class)
data = []
np.random.seed(42)

for class_name, properties in classes.items():
    label = properties['label']
    red_range, green_range, blue_range, nir_range, swir1_range = {
        'Urban': ([150, 200], [150, 200], [150, 200], [100, 150], [50, 100]),
        'Vegetation': ([30, 80], [80, 150], [20, 70], [200, 255], [150, 200]),
        'Water': ([0, 50], [0, 50], [50, 150], [10, 60], [0, 50]),
        'Agriculture': ([100, 160], [140, 200], [70, 120], [180, 240], [120, 180]),
        'Open Land': ([160, 200], [140, 180], [120, 160], [100, 140], [70, 100])
    }[class_name]
    
    for _ in range(200):  # Generate 200 samples for each class
        red = np.random.randint(*red_range)
        green = np.random.randint(*green_range)
        blue = np.random.randint(*blue_range)
        nir = np.random.randint(*nir_range)
        swir1 = np.random.randint(*swir1_range)
        data.append([red, green, blue, nir, swir1, label, class_name])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'Label', 'Class'])

# Save the dataset as CSV
output_csv = 'lulc_dataset_generated.csv'
df.to_csv(output_csv, index=False)
print(f"Generated dataset saved to: {output_csv}")

# Display first few rows
print(df.head())
