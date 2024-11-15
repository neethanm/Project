import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import tempfile
import landsat  # Import the separate landsat module

# Streamlit title
st.title("Shapefile Uploader and Viewer")

# Upload the shapefile components
st.markdown("### Upload your Shapefile")
uploaded_files = st.file_uploader(
    "Upload all files (e.g., .shp, .shx, .dbf, etc.)",
    type=["shp", "shx", "dbf", "prj"],
    accept_multiple_files=True,
)

# Check if shapefile components are uploaded
shapefile_uploaded = False
if uploaded_files:
    files = {os.path.splitext(file.name)[1]: file for file in uploaded_files}
    required_extensions = [".shp", ".shx", ".dbf"]
    if set(required_extensions).issubset(files.keys()):
        shapefile_uploaded = True

# Button for visualizing Landsat image (only enabled if shapefile is uploaded)
landsat_button = st.button("Visualize Landsat", disabled=not shapefile_uploaded)

# Set the default page in session state if it doesn't exist yet
if 'page' not in st.session_state:
    st.session_state.page = 'shapefile'

# Change page when the button is clicked
if landsat_button:
    st.session_state.page = "landsat"

# Show page based on navigation
if st.session_state.page == "landsat":
    if 'shapefile_gdf' in st.session_state:
        landsat.display_landsat_image(st.session_state.shapefile_gdf)  # Call the function from landsat.py
    else:
        st.error("Shapefile data is missing. Please upload the shapefile first.")
else:
    # If shapefile is uploaded, display it
    if shapefile_uploaded:
        try:
            # Create a temporary directory to store the uploaded files
            with st.spinner("Loading shapefile..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    for ext, file in files.items():
                        filepath = os.path.join(tmpdir, file.name)
                        with open(filepath, "wb") as f:
                            f.write(file.read())

                    # Read the shapefile using GeoPandas
                    shapefile_path = os.path.join(tmpdir, files[".shp"].name)
                    shapefile_gdf = gpd.read_file(shapefile_path)

                    # Store the GeoDataFrame in the session state
                    st.session_state.shapefile_gdf = shapefile_gdf

                    # Display basic information
                    st.write("### Shapefile Metadata")
                    st.write(shapefile_gdf)

                    # Plot the shapefile
                    st.write("### Shapefile Plot")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    shapefile_gdf.plot(ax=ax, cmap='viridis', edgecolor='black')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload the full shapefile (including .shp, .shx, and .dbf files) to enable the Landsat visualization button.")
