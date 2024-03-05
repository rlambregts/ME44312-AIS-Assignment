import pandas as pd
import json
import os

# Load the JSON data into a pandas DataFrame
with open('raw_data_rotterdam/raw_ais_data_2021_rotterdam_1609459200.0_1609545600.0.JSON') as f:
    json_data = json.load(f)

# Normalize the JSON data
data_normalized = pd.json_normalize(json_data, 'data')

# Now you can work with your DataFrame 'data_normalized'
print(data_normalized.head())  # Display the first few rows of the DataFrame