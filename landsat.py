import ee
import geemap
import geopandas as gpd
import os
# import random_forest_neetha
import streamlit as st
from zipfile import ZipFile
from pyproj import Proj, transform
import random_forest
import numpy as np
import rasterio
import matplotlib.pyplot as plt

try:
    ee.Authenticate()
    ee.Initialize(project='ee-neethanmallya111')
    st.write("Successfully authenticated and initialized Earth Engine.")
except Exception as e:
    st.write(f"Error initializing Earth Engine: {e}")

def convert_to_latlon(bounds, input_proj='EPSG:4326', output_proj='EPSG:4326'):
    # Using pyproj for coordinate transformation
    in_proj = Proj(init=input_proj)
    out_proj = Proj(init=output_proj)
    
    # Convert the corners (west, south, east, north) to lat/lon
    lon1, lat1 = transform(in_proj, out_proj, bounds[0], bounds[1])  # Bottom-left corner
    lon2, lat2 = transform(in_proj, out_proj, bounds[2], bounds[3])  # Top-right corner
    
    return [lon1, lat1, lon2, lat2]


# Function to get and filter Landsat images
def get_landsat_images(shapefile_gdf):
    # Get the bounds of the shapefile region
    bounds = shapefile_gdf.total_bounds
    st.write(f"Region bounds: {bounds}")  # Debugging line

    # Convert the bounds to lat/lon
    region_bounds = convert_to_latlon(bounds)

    # Define the region using lat/lon coordinates
    region = ee.Geometry.Rectangle([region_bounds[0], region_bounds[1], region_bounds[2], region_bounds[3]])

    # Ensure the region is in EPSG:4326 (WGS84)
    region = region.transform('EPSG:4326')

    # Print the region to verify it's correct
    st.write(f"Region geometry: {region.getInfo()}")

    # Define the years for which to collect Landsat images
    years = [2004, 2014, 2024]
    landsat_images = []

    for year in years:
        try:
            # Get the Landsat collection filtered by the region and year
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(region) \
                .filterDate(f'{year}-01-01', f'{year}-12-31')

            # Check if collection is not empty
            collection_size = collection.size().getInfo()
            st.write(f"Collection size for {year}: {collection_size}")  # Debugging line
            
            if collection_size > 0:
                # Use median composite for the year
                image = collection.median().clip(region)
                landsat_images.append((year, image))
            else:
                st.write(f"No data found for {year} within the region.")

        except Exception as e:
            st.write(f"Error processing year {year}: {e}")

    return landsat_images


# Function to export Landsat images to Google Drive
# def export_to_drive(landsat_images, region, output_dir):
#     download_links = []
    
#     for year, image in landsat_images:
#         filename = f"landsat_{year}.tif"

#         try:
#             # Export the image to Google Drive
#             export_task = ee.batch.Export.image.toDrive(
#                 image=image,
#                 description=f"landsat_{year}",
#                 fileNamePrefix=filename,
#                 region=region,
#                 scale=30,
#                 crs="EPSG:4326",
#                 fileFormat="GeoTIFF"
#             )

#             # Check for valid region
#             st.write(f"Export task for {filename} started. Checking status...")

#             export_task.start()

#             # Wait for the export task to complete
#             while export_task.active():
#                 st.write(f"Exporting {filename}...")

#             st.write(f"Export for {filename} completed successfully.")

#             # Google Drive download link
#             download_link = f"https://drive.google.com/file/d/{export_task.id()}/view"
#             download_links.append(download_link)

#         except Exception as e:
#             st.write(f"Error exporting {filename}: {e}")

#     return download_links


# def display_landsat_image(shapefile_gdf):
#     # List of Landsat image paths
#     paths = [
#         r"/Users/neethamallya/Downloads/Project-master/landsat_tif_for_testing/map_2024.tif", 
#         r"/Users/neethamallya/Downloads/Project-master/landsat_tif_for_testing/map_2014.tif", 
#         r"/Users/neethamallya/Downloads/Project-master/landsat_tif_for_testing/map_2004.tif", 
#         r"/Users/neethamallya/Downloads/Project-master/landsat_tif_for_testing/map_1994.tif"

#         # r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_1994_all_bands.tif", 
#         # r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_2004_all_bands.tif", 
#         # r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_2014_all_bands.tif", 
#         # r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_2024_all_bands.tif"
#     ]
    
#     # Create 4 columns for displaying the maps side by side
#     col1, col2, col3, col4 = st.columns(4)
    
#     for idx, path in enumerate(paths):
#         try:
#             with rasterio.open(path) as src:
#                 # Get year from path for dynamic title
#                 year = path.split('_')[-1].split('.')[0]

#                 # Normalize and create RGB composite
#                 red_band = src.read(6)  # red band
#                 green_band = src.read(5)  # green band
#                 blue_band = src.read(4)  # blue band
                
#                 def normalize(band):
#                     band = np.nan_to_num(band, nan=0)
#                     return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band) + 1e-6)

#                 red_norm = normalize(red_band)
#                 green_norm = normalize(green_band)
#                 blue_norm = normalize(blue_band)

#                 rgb_image = np.dstack((red_norm, green_norm, blue_norm)) 

#                 # Use the appropriate column to display the image
#                 if idx == 0:
#                     column = col1
#                 elif idx == 1:
#                     column = col2
#                 elif idx == 2:
#                     column = col3
#                 else:
#                     column = col4

#                 with column:
#                     # Display the image
#                     fig, ax = plt.subplots(figsize=(5, 5))
#                     ax.imshow(rgb_image)
#                     ax.set_title(f"Landsat Image {year} (RGB Composite)")
#                     ax.axis("off")
#                     st.pyplot(fig)

#                     # Dynamic button name for LULC generation
#                     lulc_button_name = f"Generate LULC for {year}"

#                     # Display the LULC button below the respective map
#                     lulc_button = st.button(lulc_button_name)

#                     if lulc_button:
#                         # Call LULC generation (adjust path if necessary)
#                         random_forest_neetha.display_lulc(path)

#         except FileNotFoundError:
#             st.error(f"File not found: {path}")
#         except Exception as e:
#             st.error(f"An error occurred with {path}: {e}")

#     ca_markov_button = st.button("Ca Markov Model")

def display_landsat_image(shapefile_gdf):
    # List of Landsat image paths
    paths = [
        r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_1994_all_bands.tif",
        r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_2004_all_bands.tif",
        r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_2014_all_bands.tif",
        r"/Users/neethamallya/Downloads/Project-master/all_bands_landsat_tif/LULC_map_2024_all_bands.tif",
    ]
    
    # Create 4 columns for displaying the maps side by side
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]

    for idx, path in enumerate(paths):
        try:
            with rasterio.open(path) as src:
                # Get year from path for dynamic title
                year = path.split('_')[-1].split('.')[0]

                # Get available band numbers
                band_count = src.count
                band_options = list(range(1, band_count + 1))

                # Use the appropriate column to display the image
                column = columns[idx]
                
                with column:
                    # User selects the bands for RGB visualization with unique keys
                    st.write(f"Select bands for Landsat {year}")
                    red_band = st.selectbox(
                        f"Red Band ({year})", band_options, index=4, key=f"red_band_{idx}_{year}"
                    )  # Default: Band 6
                    green_band = st.selectbox(
                        f"Green Band ({year})", band_options, index=3, key=f"green_band_{idx}_{year}"
                    )  # Default: Band 5
                    blue_band = st.selectbox(
                        f"Blue Band ({year})", band_options, index=2, key=f"blue_band_{idx}_{year}"
                    )  # Default: Band 4
                    
                    # Read selected bands
                    red = src.read(red_band)
                    green = src.read(green_band)
                    blue = src.read(blue_band)

                    # Normalize bands
                    def normalize(band):
                        band = np.nan_to_num(band, nan=0)
                        return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band) + 1e-6)

                    red_norm = normalize(red)
                    green_norm = normalize(green)
                    blue_norm = normalize(blue)

                    # Create RGB composite
                    rgb_image = np.dstack((red_norm, green_norm, blue_norm))

                    # Display the image
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(rgb_image)
                    ax.set_title(f"Landsat {year} - Selected Bands (R-{red_band}, G-{green_band}, B-{blue_band})")
                    ax.axis("off")
                    st.pyplot(fig)


                    # Dynamic button name for LULC generation
                    lulc_button_name = f"Generate LULC for {year}"

                    # Display the LULC button below the respective map
                    if st.button(lulc_button_name, key=f"lulc_button_{idx}_{year}"):
                        random_forest.display_lulc(path)

        except FileNotFoundError:
            st.error(f"File not found: {path}")
        except Exception as e:
            st.error(f"An error occurred with {path}: {e}")

    # Button for CA-Markov Model
    if st.button("CA-Markov Model", key="ca_markov_button"):
        st.write("CA-Markov Model button clicked.")
