import torch
import pandas as pd

# Load the edge attributes and node attributes (elevation values)
edge_attributes = torch.load('ml_for_road_safety/data/MT/MT/Edges/edge_features.pt')
node_attributes = "ml_for_road_safety/data/MT/MT/Slope/Nodes/MT_Elevation_Final.csv"
df_nodes = pd.read_csv(node_attributes)

# Coalesce the 'oneway' tensor to merge duplicate indices
oneway_coalesced = edge_attributes['oneway'].coalesce()
oneway_indices = oneway_coalesced.indices()  # Access the indices after coalescing

# Access the 'length' attribute (ensure this key exists in the edge attributes)
length = edge_attributes['length'].coalesce()
source_nodes = oneway_indices[0]
target_nodes = oneway_indices[1]

# Define thresholds
slope_cutoff = 0.3  # Maximum slope allowed for edges above 50 meters
short_length_threshold = 50  # Length threshold for additional filtering
short_length_slope_threshold = 0.2  # Threshold for setting slope = 0.0 for short edges
short_length_slope = 0.0  # Slope value for short edges with high slope
capped_slope = 0.05  # Capped slope value for steep edges (positive or negative)

# Create an empty list to store slopes
slope_values = []

print("Calculating and filtering slopes for each edge...")
for i in range(oneway_indices.size(1)):  # Iterate over all edges
    source_node = source_nodes[i].item()
    target_node = target_nodes[i].item()

    # Get the elevations for source and target nodes
    try:
        source_node_elevation = df_nodes.iloc[source_node]['elevation']
    except IndexError:
        source_node_elevation = float('nan')  # Handle missing elevation
    
    try:
        target_node_elevation = df_nodes.iloc[target_node]['elevation']
    except IndexError:
        target_node_elevation = float('nan')  # Handle missing elevation

    # Compute elevation difference
    elevation_difference = target_node_elevation - source_node_elevation

    # Ensure we correctly access the length for the current edge
    try:
        edge_length = length.values()[i].item()
    except IndexError:
        edge_length = float('nan')  # Handle missing length

    # Compute slope (avoid division by zero)
    if edge_length and edge_length != 0:
        slope = elevation_difference / edge_length
    else:
        slope = float('nan')  # Handle invalid length values

    # Apply slope rules
    if edge_length < short_length_threshold and abs(slope) > short_length_slope_threshold:
        # Automatically set slope = 0.0 for short edges with slope > 0.2
        print(f"Setting slope for short edge {source_node} -> {target_node} (length {edge_length}): {slope} -> {short_length_slope}")
        slope = short_length_slope
    elif abs(slope) > slope_cutoff:
        # Cap slope for steep edges
        capped_value = capped_slope if slope > 0 else -capped_slope
        print(f"Capping slope for steep edge {source_node} -> {target_node}: {slope} -> {capped_value}")
        slope = capped_value

    slope_values.append(slope)

slope_values = [0.0 if torch.isnan(torch.tensor(s)) else s for s in slope_values]

# Convert slope values to a tensor
slope_tensor = torch.tensor(slope_values)
edge_attributes['slope'] = torch.sparse_coo_tensor(
    indices=oneway_indices,  
    values=slope_tensor,    
    size=length.size()     
)

# Save the updated edge attributes
torch.save(edge_attributes, 'ml_for_road_safety/data/MT/MT/Edges/edge_features_with_slope.pt')

print("Slope attribute added and saved to 'edge_features_with_slope.pt'.")
