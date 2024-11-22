from pyrosm import OSM
import geopandas as gpd
import osmnx as ox
import numpy as np
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# Load OSM data and extract road network
osm = OSM(r"osm_data/karnataka.osm.pbf")
gdf_roads = osm.get_network(network_type="driving")  # Specify road type

# Load region of interest (ROI)
shapefile_path = r"shapefiles_for_testing/Manglore.shp"
region_of_interest = gpd.read_file(shapefile_path)

# Ensure both datasets have the same CRS
gdf_roads = gdf_roads.to_crs(region_of_interest.crs)

# Crop road network to ROI
cropped_roads = gpd.overlay(gdf_roads, region_of_interest, how='intersection')

# Define rasterization parameters
raster_width, raster_height = 500, 500
bounds = cropped_roads.total_bounds  # [west, south, east, north]
transform = from_origin(bounds[0], bounds[3], (bounds[2] - bounds[0]) / raster_width, 
                         (bounds[3] - bounds[1]) / raster_height)

# Create a rasterized mask of the road network
roads_geom = cropped_roads.geometry
road_mask = geometry_mask(
    geometries=roads_geom,
    transform=transform,
    invert=True,
    out_shape=(raster_height, raster_width)
)

# Calculate the distance map
distance_map = distance_transform_edt(~road_mask)

# Save the distance map as a GeoTIFF file
output_file = r"osm_results\road_network.tif"
with rasterio.open(
    output_file,
    'w',
    driver='GTiff',
    height=raster_height,
    width=raster_width,
    count=1,
    dtype=distance_map.dtype,
    crs=region_of_interest.crs,
    transform=transform
) as dst:
    dst.write(distance_map, 1)

# Visualize the distance map
plt.imshow(distance_map, cmap='viridis')
plt.colorbar(label='Distance to nearest road (m)')
plt.title('Distance from Roads')
plt.show()

print(f"Road network distance map saved as {output_file}")
