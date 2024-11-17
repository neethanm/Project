import streamlit as st
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Streamlit app setup
st.title("Clip Raster (LULC) with Shapefile")

# File paths
raster_file_path = "temlulc\LULC_2024.tif"  # Replace with your raster file path
shapefile_path = "temp_shapefile\Manglore.shp"    # Replace with your shapefile path

try:
    # Load the raster file
    with rasterio.open(raster_file_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        st.write(f"Raster CRS: {raster_crs}")
        st.write(f"Raster Bounds: {raster_bounds}")

    # Load the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Check if shapefile has CRS
    if shapefile.crs is None:
        st.error("Shapefile does not have a CRS. Please check the shapefile.")
    else:
        st.write(f"Shapefile CRS: {shapefile.crs}")

    # Reproject shapefile to match raster CRS (EPSG:4326)
    if shapefile.crs != raster_crs:
        shapefile = shapefile.to_crs(raster_crs)
        st.write("Reprojected shapefile to match raster CRS.")

    # Check bounds overlap after reprojection
    shapefile_bounds = shapefile.total_bounds  # (minx, miny, maxx, maxy)
    st.write(f"Shapefile Bounds: {shapefile_bounds}")

    if not (
        shapefile_bounds[0] < raster_bounds[2] and  # minx < raster maxx
        shapefile_bounds[2] > raster_bounds[0] and  # maxx > raster minx
        shapefile_bounds[1] < raster_bounds[3] and  # miny < raster maxy
        shapefile_bounds[3] > raster_bounds[1]      # maxy > raster miny
    ):
        st.error("Error: Shapefile does not overlap raster.")
    else:
        st.write("Shapefile and raster overlap detected.")

    # Validate shapefile geometries
    shapefile = shapefile[shapefile.is_valid]
    if shapefile.empty:
        raise ValueError("No valid geometries found in the shapefile.")

    # Extract geometries from shapefile
    geometries = shapefile.geometry.values

    # Clip the raster using shapefile
    with rasterio.open(raster_file_path) as src:
        clipped_data, clipped_transform = mask(src, geometries, crop=True)
        profile = src.profile

    # Update raster profile for the clipped output
    profile.update({
        "height": clipped_data.shape[1],
        "width": clipped_data.shape[2],
        "transform": clipped_transform
    })

    # Define a custom colormap for LULC classes
    lulc_colors = [
        (0.8, 0.9, 0.4),    # Class 1: Light Green
        (0.9, 0.1, 0.1),    # Class 2: Urban (Red)
        (0.4, 0.6, 0.2),    # Class 3: Dark Green
        (0.9, 0.6, 0.2),    # Class 4: Orange
        (0.6, 0.4, 0.2),    # Class 5: Brown
        (0.2, 0.4, 0.6),    # Class 6: Blue
    ]
    cmap = ListedColormap(lulc_colors)

    # Display the clipped raster
    st.write("Clipped LULC Map:")
    fig, ax = plt.subplots()
    cax = ax.imshow(clipped_data[0], cmap=cmap)  # [0] because mask returns a 3D array
    ax.axis("off")
    fig.colorbar(cax, ax=ax, orientation="vertical", label="LULC Classes")
    st.pyplot(fig)

    # Optional: Save the clipped raster
    save_option = st.checkbox("Save clipped raster to file?")
    if save_option:
        output_path = "clipped_lulc.tif"  # Change the file name as needed
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped_data)
        st.write(f"Clipped raster saved to: {output_path}")

except Exception as e:
    st.error(f"An error occurred: {e}")
