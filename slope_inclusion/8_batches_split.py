import pandas as pd
import os

# Configuration
input_file = "ml_for_road_safety/data/MT/MT/Nodes/node_features_2016_1.csv"
output_dir = "ml_for_road_safety/data/MT/MT/Slope/Nodes/Node_Batches"
batch_size = 1000  # Number of rows per file

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the input file
data = pd.read_csv(input_file)

# Get total number of rows
total_rows = data.shape[0]

# Split the data into chunks
for i in range(0, total_rows, batch_size):
    # Create a batch
    batch = data.iloc[i:i + batch_size]
    # Save to a new CSV file
    output_file = os.path.join(output_dir, f"MT_{i // batch_size + 1}.csv")
    batch.to_csv(output_file, index=False)

print(f"Data successfully split into batches of {batch_size} rows and saved in {output_dir}.")
