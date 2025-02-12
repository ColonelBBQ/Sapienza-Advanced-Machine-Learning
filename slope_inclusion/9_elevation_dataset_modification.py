import pandas as pd
import os
import requests
import json
import time

# Configuration
input_dir = r"ml_for_road_safety\data\MT\MT\Slope\Nodes\Node_Batches"
output_dir = r"ml_for_road_safety\data\MT\MT\Slope\Nodes\Node_Elevation_Batches"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open-Elevation API URL1
url = "https://api.open-elevation.com/api/v1/lookup"

# Function to get elevation for a list of coordinates
def get_elevation(coords):
    payload = {"locations": [{"latitude": lon, "longitude": lat} for lat, lon in coords]}
    headers = {"Content-Type": "application/json"}
    
    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error: {response.status_code}")
        return None

# Function to process a specific batch
def process_batch(batch_name):
    input_file = os.path.join(input_dir, batch_name)
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"File {input_file} does not exist.")
        return False
    
    # Read the batch file
    data = pd.read_csv(input_file)
    
    # Extract latitude and longitude columns (assuming they are named 'lat' and 'lon')
    coords = data[['lat', 'lon']].values.tolist()
    
    # Get elevation data for these coordinates
    elevations = get_elevation(coords)
    
    if elevations:
        # Add elevation data to the dataframe
        data['elevation'] = [elevation['elevation'] for elevation in elevations]
        
        # Save the updated batch with elevation data
        output_file = os.path.join(output_dir, f"Elevation_{batch_name}")
        data.to_csv(output_file, index=False)
        print(f"Elevation data added and saved to {output_file}")
        return True
    else:
        print(f"Error occurred while processing {batch_name}.")
        return False

# Start processing all batches automatically
start_batch = input("Enter the starting batch number (or leave blank to start from the beginning): ")

# Handle starting from a specific batch
if start_batch:
    try:
        start_batch = int(start_batch)
    except ValueError:
        print("Invalid batch number, starting from the beginning.")
        start_batch = 1
else:
    start_batch = 1

# List of all batch files in the directory
batch_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Sort batch files numerically (assuming filenames like NV_1.csv, NV_2.csv, etc.)
batch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# Process batches from the specified starting batch
for i, batch_name in enumerate(batch_files[start_batch - 1:], start=start_batch):
    print(f"Processing batch {batch_name}...")
    
    # Try processing the batch, if it fails, break out of the loop
    if not process_batch(batch_name):
        print(f"Error processing {batch_name}. You can manually start from batch {i}.")
        break
    
    # Sleep for 5 seconds between requests to avoid hitting the API rate limit
    time.sleep(5)

print("Processing complete.")
