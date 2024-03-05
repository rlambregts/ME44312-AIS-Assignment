import os
import json
import pandas as pd

data_path = r'.\Data'

# Initialize an empty list to store parsed JSON data
all_data = []

# Iterate over files in the directory
for file_name in os.listdir(data_path):
    # Construct full file path
    file_path = os.path.join(data_path, file_name)
    
    # Check if the path is a file (not a directory)
    if os.path.isfile(file_path):
        # Open and read the file
        with open(file_path) as f:
            # Load JSON data from the file
            json_data = json.load(f)
            # Append loaded data to the list
            all_data.append(json_data)

# Normalize the JSON data
data_normalized = pd.json_normalize(all_data, 'data')

# Now you can work with your DataFrame 'data_normalized'
print(data_normalized.head())  # Display the first few rows of the DataFrame
