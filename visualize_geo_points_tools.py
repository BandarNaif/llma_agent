
from langchain.tools import tool
import pandas as pd
import folium
from itertools import cycle
@tool
def visualize_geo_points(path:str, category_column:str, lat_column='lat', lon_column='lon', output_file='map.html'):
    """
    Visualize geospatial points on a map with category-based pin colors.
    
    Parameters:
        path (str): Path to CSV file.
        category_column (str): Column name for categories (e.g., 'type').
        lat_column (str): Column name for latitude values.
        lon_column (str): Column name for longitude values.
        output_file (str): Name of the output HTML file.
    
    Returns:
        str: Path to the generated map HTML file.
    """
    df = pd.read_csv(path)
    # Predefined color palette
    predefined_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue'
    ]

    # Assign colors dynamically to unique categories
    unique_categories = df[category_column].unique()
    color_cycle = cycle(predefined_colors)  # Cycle through colors if categories > colors
    color_map = {category: next(color_cycle) for category in unique_categories}

    # Create a Folium map centered on the mean location
    center_lat = df[lat_column].mean()
    center_lon = df[lon_column].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    # Add points to the map with dynamic colors
    for _, row in df.iterrows():
        # Generate a dynamic popup text
        popup_content = "<br>".join(
            [f"{col}: {row[col]}" for col in df.columns if col not in [lat_column, lon_column]]
        )

        folium.Marker(
            location=[row[lat_column], row[lon_column]],
            popup=popup_content,
            tooltip=row[category_column],
            icon=folium.Icon(color=color_map.get(row[category_column], 'gray'))  # Default to 'gray'
        ).add_to(m)

    # Save map to file
    m.save(output_file)
    return f"Map saved to {output_file}"
