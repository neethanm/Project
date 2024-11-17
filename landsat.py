import ee
import geemap
import geopandas as gpd
import os
import random_forest
import streamlit as st
from zipfile import ZipFile
from pyproj import Proj, transform
# import random_forest
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
def export_to_drive(landsat_images, region, output_dir):
    download_links = []
    
    for year, image in landsat_images:
        filename = f"landsat_{year}.tif"

        try:
            # Export the image to Google Drive
            export_task = ee.batch.Export.image.toDrive(
                image=image,
                description=f"landsat_{year}",
                fileNamePrefix=filename,
                region=region,
                scale=30,
                crs="EPSG:4326",
                fileFormat="GeoTIFF"
            )

            # Check for valid region
            st.write(f"Export task for {filename} started. Checking status...")

            export_task.start()

            # Wait for the export task to complete
            while export_task.active():
                st.write(f"Exporting {filename}...")

            st.write(f"Export for {filename} completed successfully.")

            # Google Drive download link
            download_link = f"https://drive.google.com/file/d/{export_task.id()}/view"
            download_links.append(download_link)

        except Exception as e:
            st.write(f"Error exporting {filename}: {e}")

    return download_links


# Function to display the Landsat images
def display_landsat_image(shapefile_gdf):
    # # Get Landsat images based on shapefile region
    # landsat_images = get_landsat_images(shapefile_gdf)
    
    # if landsat_images:
    #     # Export the Landsat images to Google Drive
    #     download_links = export_to_drive(landsat_images, shapefile_gdf, "downloads")
        
    #     if download_links:
    #         # Provide download links for the exported files
    #         for link in download_links:
    #             st.write(f"Download from Google Drive: {link}")
    # else:
    #     st.write("No Landsat images available for the selected region.")
    landsat_2024_path = r"C:\Users\91984\Downloads\LandsatImageExport.tif"

    try:
        with rasterio.open(landsat_2024_path) as src:
            st.write("Image Metadata:")
            st.json(src.meta)

            if src.count < 3:
                st.error("This file does not have enough bands for RGB visualization.")
            else:
                red_band = src.read(4) # red band
                green_band = src.read(3) # green band
                blue_band = src.read(2) # blue band
                st.write("Band Statistics:")
                st.write(f"Red Band - Min: {red_band.min()}, Max: {red_band.max()}")
                st.write(f"Green Band - Min: {green_band.min()}, Max: {green_band.max()}")
                st.write(f"Blue Band - Min: {blue_band.min()}, Max: {blue_band.max()}")
                # Print a portion of the red band for debugging
                st.write("Red Band Sample Values:")
                st.write(red_band[0:5, 0:5])


                def normalize(band):
                    band = np.nan_to_num(band, nan=-0)
                    return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band) + 1e-6)
            
            red_norm = normalize(red_band)
            green_norm = normalize(green_band)
            blue_norm = normalize(blue_band)

            # rgb composite
            rgb_image = np.dstack((red_norm, green_norm, blue_norm)) 

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_image)
            ax.set_title("Landsat Image 2024 (RGB Composite)")
            ax.axis("off")
            st.pyplot(fig)

            lulc_button = st.button("Generate LULC")

            if lulc_button:
                random_forest.display_lulc(landsat_2024_path)

    except FileNotFoundError:
        st.error(f"File not found: {landsat_2024_path}")
    except Exception as e:
        st.error(f"An error occurred: {e}")



# # Streamlit file uploader for shapefile
# shapefile_gdf = st.file_uploader("Upload a Shapefile", type=[".shp", ".zip"])

# if shapefile_gdf is not None:
#     # Load shapefile using GeoPandas
#     with ZipFile(shapefile_gdf) as zf:
#         shapefile_gdf = gpd.read_file(zf.open(zf.namelist()[0]))

#     # Display the first few rows of the shapefile for confirmation
#     st.write(shapefile_gdf.head())
    
#     # Display Landsat image based on the uploaded shapefile
#     display_landsat_image(shapefile_gdf)
