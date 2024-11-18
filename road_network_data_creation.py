# plan for now

# obtain road network from somewhere (probably gov datasets)

# extract the road network and simpify it

# create raster grid that covers the region

#calculate distance from roads - rasterize then use a distance algorithm (most preferably euclidean itself)

# visualization of the distance map - matplotlib

from pyrosm import OSM
import geopandas as gpd
import osmnx as ox
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask

osm = OSM(r"osm_data\karnataka.osm.pbf")

gdf_roads = osm.get_network()

print(gdf_roads.head())

shapefile_path = r"shapefiles_for_testing\Manglore.shp"
region_of_interest = gpd.read_file(shapefile_path)

print(region_of_interest.head())

# cropping
gdf_roads = gdf_roads.to_crs(region_of_interest.crs)

cropped_roads = gpd.overlay(gdf_roads, region_of_interest, how='intersection')

print(cropped_roads.head())

# converting to a graph
graph = ox.graph_from_gdfs(cropped_roads)
gdf_nodes, gdf_edges = ox.graph_to_gdf(graph)

gdf_edges.to_file(r"osm_results\cropped_roads.shp")
gdf_nodes.to_file(r"osm_results\cropped_nodes.shp")

# rasteriszing of the roadd network (roads with 1 and non-roads with 0)
roads_geom = cropped_roads.geometry

raster_width, raster_height = 500, 500
transform = rasterio.transform.from_origin(west = cropped_roads.bounds[0], north=cropped_roads.bounds[3], xsize=100, ysize=100)

roads_mask = geometry_masl(roads_geom, transform=transform, invert=True, out_shape=(raster_height, raster_width))

distance_map = distance_transform_edit(road_mask)

plt.imshow(distance_map, cmap='viridis')
plt.colorbar(label='Distance to nearest road (m)')
plt.title('Distance from roads')
plt.show()