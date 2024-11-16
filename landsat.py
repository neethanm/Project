import ee
import geemap
import os
import streamlit as st
import geopandas as gpd
from zipfile import ZipFile

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
    years = [2020, 2014, 2024]
    landsat_images = []

    for year in years:
        # Landsat collection for the given year (use surface reflectance data)
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')  # Surface reflectance data
        collection = collection.filterBounds(region).filterDate(f'{year}-01-01', f'{year}-12-31')

        # Median composite for the year
        image = collection.median().clip(region)
        landsat_images.append(image)

    return landsat_images

def export_and_download_landsat(landsat_images, region, output_dir):
    download_links = []
    for idx, image in enumerate(landsat_images):
        filename = f"landsat_{2020+idx}.tif"
        file_path = os.path.join(output_dir, filename)

        try:
            # Check if the directory exists, and create it if necessary
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            # Check if file already exists, and if not, export
            if not os.path.exists(file_path):
                print(f"Exporting {filename} to {file_path}")

                # Check if the image has valid bands (non-empty)
                if image.bandNames().size().getInfo() == 0:
                    print(f"Image for {2020+idx} is empty, skipping.")
                    continue

                # Add CRS and scale to the export function
                export_task = ee.batch.Export.image.toDrive(
                    image=image,
                    description=f"Landsat_{2020+idx}",
                    scale=30,  # Use a standard Landsat scale
                    region=region,  # Explicit region parameter
                    crs="EPSG:4326",  # WGS84 projection (longitude/latitude)
                    fileNamePrefix=filename
                )
                export_task.start()
                download_links.append(file_path)
                print(f"Successfully exported {filename}")
            else:
                print(f"File already exists: {file_path}")

        except Exception as e:
            print(f"Error exporting {filename}: {e}")

    return download_links

def display_landsat_image(shapefile_gdf):
    st.subheader("Landsat Images from 2020, 2014, and 2024")

    # Get the Landsat images
    landsat_images = get_landsat_images(shapefile_gdf)
    
    # Extract region for export
    bounds = shapefile_gdf.total_bounds
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

    # Create a subdirectory for downloads in the project directory
    output_dir = os.path.join(os.getcwd(), "downloads")
    
    # Export and download the Landsat images
    st.info("Downloading Landsat TIFF files...")
    tiff_files = export_and_download_landsat(landsat_images, region, output_dir)

    # Display download links
    st.success("Download complete! Here are the files:")
    for tiff_file in tiff_files:
        # Provide a clickable link to the downloaded files
        st.write(f"[Download {os.path.basename(tiff_file)}](file://{tiff_file})")

    # Display TIFF files on a map using geemap
    map = geemap.Map(center=[shapefile_gdf.geometry.centroid.y.mean(), shapefile_gdf.geometry.centroid.x.mean()], zoom=10)

    # Add raster layers to the map after ensuring files are downloaded
    for tiff_file in tiff_files:
        try:
            # Check if file exists before adding it to the map
            if os.path.exists(tiff_file):
                map.add_raster(tiff_file, layer_name=os.path.basename(tiff_file))
            else:
                print(f"File {tiff_file} not found, skipping...")
        except Exception as e:
            print(f"Error adding raster {tiff_file} to map: {e}")
    
    # Display the map in Streamlit
    map.to_streamlit(height=500)

# Example usage in Streamlit
if __name__ == "__main__":
    # Assuming shapefile_gdf is loaded as a GeoDataFrame containing your region of interest
    shapefile_file = st.file_uploader("Upload a Shapefile", type=[".shp", ".zip"])

    if shapefile_file is not None:
        # Handle ZIP files (if applicable)
        if shapefile_file.name.endswith('.zip'):
            with ZipFile(shapefile_file) as zf:
                zf.extractall("/tmp/shapefile")
            shapefile_path = "/tmp/shapefile"
        else:
            shapefile_path = shapefile_file.name
        
        shapefile_gdf = gpd.read_file(shapefile_path)
        display_landsat_image(shapefile_gdf)
