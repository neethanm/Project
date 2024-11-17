import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import tempfile
import landsat 
import alternate_landsat

st.title("Shapefile Uploader and Viewer")

st.markdown("### Upload your Shapefile")
uploaded_files = st.file_uploader(
    "Upload all files (e.g., .shp, .shx, .dbf, etc.)",
    type=["shp", "shx", "dbf", "prj"],
    accept_multiple_files=True,
)

shapefile_uploaded = False
if uploaded_files:
    files = {os.path.splitext(file.name)[1]: file for file in uploaded_files}
    required_extensions = [".shp", ".shx", ".dbf"]
    if set(required_extensions).issubset(files.keys()):
        shapefile_uploaded = True

landsat_button = st.button("Visualize Landsat", disabled=not shapefile_uploaded)

if 'page' not in st.session_state:
    st.session_state.page = 'shapefile'


if landsat_button:
    st.session_state.page = "landsat"

if st.session_state.page == "landsat":
    if 'shapefile_gdf' in st.session_state:
        alternate_landsat.display_landsat_image(st.session_state.shapefile_gdf)  # Call the function from landsat.py
    else:
        st.error("Shapefile data is missing. Please upload the shapefile first.")
else:
    if shapefile_uploaded:
        try:
            with st.spinner("Loading shapefile..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    for ext, file in files.items():
                        filepath = os.path.join(tmpdir, file.name)
                        with open(filepath, "wb") as f:
                            f.write(file.read())

                    shapefile_path = os.path.join(tmpdir, files[".shp"].name)
                    shapefile_gdf = gpd.read_file(shapefile_path)

                    st.session_state.shapefile_gdf = shapefile_gdf

                    st.write("### Shapefile Metadata")
                    st.write(shapefile_gdf)

                    st.write("### Shapefile Plot")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    shapefile_gdf.plot(ax=ax, cmap='viridis', edgecolor='black')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload the full shapefile (including .shp, .shx, and .dbf files) to enable the Landsat visualization button.")
