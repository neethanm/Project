import ee
import geemap
import streamlit as st

# Authenticate and initialize Earth Engine
try:
    ee.Authenticate()
except Exception as e:
    print(f"Authentication failed: {e}")

# Initializing Earth Engine
try:
    ee.Initialize(project='ee-neethanmallya111')
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")

def get_landsat_images(shapefile_gdf):
    # Extract bounding box from GeoDataFrame and convert to a list
    bounds = shapefile_gdf.total_bounds
    bounds_list = [bounds[0], bounds[1], bounds[2], bounds[3]]  # [minx, miny, maxx, maxy]
    
    # Create an Earth Engine rectangle using the bounds
    region = ee.Geometry.Rectangle(bounds_list)

    # Adjusted years for Landsat 8 (starting from 2013)
    years = [2004, 2014, 2024]
    landsat_images = []

    for year in years:
        # Landsat collection for the given year (use surface reflectance data)
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')  # Surface reflectance data
        collection = collection.filterBounds(region).filterDate(f'{year}-01-01', f'{year}-12-31')

        # Median composite for the year
        image = collection.median().clip(region)

        # Visualization parameters for Landsat 8 (RGB bands)
        vis_params = {
            'min': 0,
            'max': 3000,
            'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # Corrected RGB bands
        }
        landsat_images.append(image)

    return landsat_images

def display_landsat_image(shapefile_gdf):
    st.subheader("Landsat Images from 2004, 2014, and 2024")

    landsat_images = get_landsat_images(shapefile_gdf)

    # Create a geemap Map object to display the images
    map = geemap.Map(center=[shapefile_gdf.geometry.centroid.y, shapefile_gdf.geometry.centroid.x], zoom=10)

    # Add each Landsat image to the map
    for i, image in enumerate(landsat_images):
        # Add image to the map with a label
        map.addLayer(image, {'min': 0, 'max': 3000, 'bands': ['SR_B4', 'SR_B3', 'SR_B2']}, f"Landsat {2004 + (i*10)}")

    # Display the map using Streamlit's to_streamlit() method
    try:
        map.to_streamlit(height=500)
    except Exception as e:
        st.error(f"Error displaying the map: {e}")
