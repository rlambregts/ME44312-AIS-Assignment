import os
import json
import pandas as pd

data_path = r'C:\Users\julia\OneDrive\Documenten\TU Delft\1. MSc TIL\Machine learning\ME44312-AIS-Assignment\Data'

# Initialize lists to store latitudes and longitudes
latitudes = []
longitudes = []

# Iterate over files in the directory
for file_name in os.listdir(data_path):
    # Construct full file path
    file_path = os.path.join(data_path, file_name)
    
    # Check if the path is a file (not a directory)
    if os.path.isfile(file_path):
        try:
            # Open and read the file
            with open(file_path, 'r') as f:
                # Parse JSON data
                parsed_data = json.load(f)
                # Check if 'navigation' key exists
                if 'navigation' in parsed_data and 'location' in parsed_data['navigation']:
                    # Append latitude and longitude to lists
                    latitudes.append(parsed_data['navigation']['location']['lat'])
                    longitudes.append(parsed_data['navigation']['location']['long'])
        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")

# Create a DataFrame with latitudes and longitudes
df_locations = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})

# Display the DataFrame
print(df_locations)
