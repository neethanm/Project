import rasterio
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# Define color codes for 5 land use classes
red = np.array([255, 0, 0])  # Urban
dgreen = np.array([0, 128, 0])  # Dense Vegetation
blue = np.array([0, 0, 255])  # Water
yellow = np.array([255, 255, 0])  # Barren land
lgreen = np.array([0, 255, 0])  # Light Vegetation
colors = [red, yellow, blue, dgreen, lgreen]  # Only 5 classes

def color(x):
    if((x == blue).all()):
        return 0
    elif((x == red).all()):
        return 1
    elif((x == lgreen).all()):
        return 2
    elif((x == dgreen).all()):
        return 3
    elif((x == yellow).all()):
        return 4
    else:
        return 5  # Default case for other values, can be avoided if strictly 5 classes

def colorRet(x):
    if(x == 0):
        return blue  # Water
    elif(x == 1):
        return red  # Urban
    elif(x == 2):
        return lgreen  # Light Vegetation
    elif(x == 3):
        return dgreen  # Dense Vegetation
    elif(x == 4):
        return yellow  # Barren Land
    else:
        return red  # Default case for other values, can be avoided if strictly 5 classes

class markovImg:
    def __init__(self, data):
        self.npimg = data
        self.imgH = self.npimg.shape[0]
        self.imgL = self.npimg.shape[1]
        self.nbr = np.full((self.imgH, self.imgL), -1)
        for i in range(self.imgH):
            for j in range(self.imgL):
                self.nbr[i][j] = self.neighbor(i, j)

    @classmethod
    def link(cls, filepath):
        # Reading a .tif file using rasterio
        with rasterio.open(filepath) as src:
            num_bands = src.count
            if num_bands == 1:
                img = src.read(1)
                img = np.expand_dims(img, axis=-1)
            else:
                img = src.read([1, 2, 3])  # Assuming the input image has 3 bands (RGB)
                img = np.moveaxis(img, 0, -1)  # Reordering the dimensions to (height, width, 3)
        return markovImg(img)

    def pxl(self, x, y):
        return self.npimg[x][y]

    def pxlc(self, x, y):
        return color(self.npimg[x][y])

    def h(self):
        return self.imgH

    def l(self):
        return self.imgL

    def neighbor(self, x, y):
        try:
            nlist = [(self.pxlc(x + 1, y + 1)), (self.pxlc(x, y + 1)), 
                     (self.pxlc(x - 1, y + 1)), (self.pxlc(x + 1, y)),
                     (self.pxlc(x - 1, y)), (self.pxlc(x + 1, y - 1)),
                     (self.pxlc(x, y - 1)), (self.pxlc(x - 1, y - 1))]
            n2 = [i for i in nlist if i > 4]
            if not n2 and self.pxlc(x, y) < 5:
                n = nlist.count(0) + 10 * nlist.count(1) + 100 * nlist.count(2) + 1000 * nlist.count(3) + 10000 * nlist.count(4) + 100000 * color(self.pxl(x, y))
                return n
            else:
                return -1
        except:
            return -1

# Function to generate the transition matrix for temporal modeling
def generate_transition_matrix(img1, img2):
    # img1 and img2 represent two different time steps (e.g., two years)
    transition_matrix = np.zeros((5, 5), dtype=float)
    
    # Loop through each pixel and calculate transitions from img1 to img2
    for i in range(img1.h()):
        for j in range(img1.l()):
            if color(img1.pxl(i, j)) < 5 and color(img2.pxl(i, j)) < 5:
                transition_matrix[color(img1.pxl(i, j))][color(img2.pxl(i, j))] += 1
    
    # Normalize the matrix to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    for i in range(5):
        if row_sums[i] != 0:
            transition_matrix[i] /= row_sums[i]
    
    return transition_matrix

# Spatial and Temporal modeling function (Markov Prediction)
def markovPredict(img1, img2, output_path):
    transition_matrix = generate_transition_matrix(img1, img2)
    
    # Now use the transition matrix to predict the next land use class based on temporal information
    predicted_image = np.full((img1.h(), img1.l(), 3), 255, dtype=np.uint8)  # Initialize the image
    for i in range(img1.h()):
        for j in range(img1.l()):
            if color(img1.pxl(i, j)) < 5 and color(img2.pxl(i, j)) < 5:
                x = float(random.randint(0, 1000000)) / 1000000  # Random value for probabilistic transition
                y = color(img2.pxl(i, j))
                for k in range(5):
                    if x <= transition_matrix[y][k]:
                        predicted_image[i][j] = colorRet(k)
                        break

    # Saving the output as a .tif file
    with rasterio.open(filename, 'w', driver='GTiff', count=3, dtype='uint8', 
                       width=img1.l(), height=img1.h(), crs='EPSG:4326', 
                       transform=img2.transform) as dst:
        for i in range(3):
            dst.write(predicted_image[..., i], i + 1)
    
    # Display the output image
    img = Image.fromarray(predicted_image, 'RGB')
    img.show()

# You can now call this function with your LULC .tif files
img1 = markovImg.link(r"random_lulcs_for_testing\LULC_2018.tif")
img2 = markovImg.link(r"random_lulcs_for_testing\LULC_2024.tif")
markovPredict(img1, img2, r"predicted_output.tif")
