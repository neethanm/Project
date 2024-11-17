import numpy as np
import pandas as pd
import random

# Define the ranges for each class (based on general spectral properties)
lulc_classes = ['Urban', 'Vegetation', 'Water', 'Agriculture', 'Open Land']

# Define ranges for spectral values (Red, Green, Blue, NIR, SWIR1)
# These ranges are approximate and should be adjusted based on actual Landsat 8 data
ranges = {
    'Urban': {
        'Red': (100, 200),
        'Green': (80, 180),
        'Blue': (50, 150),
        'NIR': (20, 100),
        'SWIR1': (50, 100)
    },
    'Vegetation': {
        'Red': (20, 100),
        'Green': (30, 120),
        'Blue': (20, 100),
        'NIR': (100, 200),
        'SWIR1': (60, 120)
    },
    'Water': {
        'Red': (0, 50),
        'Green': (0, 50),
        'Blue': (0, 50),
        'NIR': (0, 30),
        'SWIR1': (0, 30)
    },
    'Agriculture': {
        'Red': (60, 150),
        'Green': (60, 140),
        'Blue': (50, 120),
        'NIR': (50, 150),
        'SWIR1': (40, 100)
    },
    'Open Land': {
        'Red': (70, 160),
        'Green': (60, 150),
        'Blue': (50, 130),
        'NIR': (30, 100),
        'SWIR1': (50, 110)
    }
}

# Function to generate random data points based on the ranges for each class
def generate_class_data(class_name, num_samples):
    data = []
    for _ in range(num_samples):
        # Randomly select values for each band within the defined range for the selected class
        red = random.randint(ranges[class_name]['Red'][0], ranges[class_name]['Red'][1])
        green = random.randint(ranges[class_name]['Green'][0], ranges[class_name]['Green'][1])
        blue = random.randint(ranges[class_name]['Blue'][0], ranges[class_name]['Blue'][1])
        nir = random.randint(ranges[class_name]['NIR'][0], ranges[class_name]['NIR'][1])
        swir1 = random.randint(ranges[class_name]['SWIR1'][0], ranges[class_name]['SWIR1'][1])
        
        # Append the data point with the class label (as a number)
        data.append([red, green, blue, nir, swir1, class_name])
    
    return data

# Number of samples per class
num_samples_per_class = 1000

# Generate data for each LULC class
data = []
for class_name in lulc_classes:
    class_data = generate_class_data(class_name, num_samples_per_class)
    data.extend(class_data)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'Class'])

# Map class names to numerical labels (0=Urban, 1=Vegetation, 2=Water, 3=Agriculture, 4=Open Land)
class_mapping = {'Urban': 0, 'Vegetation': 1, 'Water': 2, 'Agriculture': 3, 'Open Land': 4}
df['Class'] = df['Class'].map(class_mapping)

# Save the dataset to a CSV file
df.to_csv("lulc_dataset_updated.csv", index=False)

# Display the first few rows of the generated dataset
print(df.head())
