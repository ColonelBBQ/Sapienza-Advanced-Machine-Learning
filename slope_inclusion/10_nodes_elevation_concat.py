import pandas as pd
import os
import re  # To extract the numeric part from the filename

# Configuration
input_dir = r"ml_for_road_safety\data\MT\MT\Slope\Nodes\Node_Elevation_Batches"
output_file = r"ml_for_road_safety\data\MT\MT\Slope\Nodes\MT_Elevation_Final.csv"

# List all batch files in the Elevation_Batches directory
batch_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Sort batch files by the numeric part after 'MT_' (e.g., 'MT_1.csv', 'MT_2.csv', ...)
batch_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

# Initialize an empty list to hold all DataFrames
all_data = []

# Loop over each batch file and read it into a DataFrame
for batch_file in batch_files:
    file_path = os.path.join(input_dir, batch_file)
    print(f"Reading {batch_file}...")
    
    # Read the batch CSV file into a DataFrame
    batch_data = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    all_data.append(batch_data)

# Concatenate all the DataFrames into a single DataFrame
final_data = pd.concat(all_data, ignore_index=True)

final_data.to_csv(output_file, index=False)
print(f"Final combined file saved as: {output_file}")
