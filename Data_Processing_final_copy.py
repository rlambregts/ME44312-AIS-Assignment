#!/usr/bin/env python
# coding: utf-8

# In[65]:


import os
import json
import pandas as pd
import matplotlib.pyplot  as plt
import folium
from folium.plugins import MarkerCluster
import mplleaflet

AIS_path = r'.\Data'
AIS_data = []

# Iterate over files in the directory
for file_name in os.listdir(AIS_path):
    # Construct full file path
    file_path = os.path.join(AIS_path, file_name)
    
    # Check if the path is a file (not a directory)
    if os.path.isfile(file_path):
        # Open and read the file
        with open(file_path) as f:
            # Load JSON data from the file
            json_data = json.load(f)
            # Append loaded data to the list
            AIS_data.append(json_data)

# Normalize the JSON data
data_normalized = pd.json_normalize(AIS_data, 'data')

# Now you can work with your DataFrame 'data_normalized'
data_normalized


# In[66]:


# data_normalized['vessel.type'].unique()
data_normalized['vessel.name'].unique()
# data_normalized['navigation.status'].unique()
# data_normalized['navigation.destination.name'].unique()
# data_normalized['navigation.location.long'].unique()

data_normalized['vessel.name'].value_counts()


# In[67]:


from shapely.geometry import Point, Polygon

# Define the coordinates of the polygon
polygon_coords = [(51.549,3.080), (51.334,3.384), (51.317,4.198), (51.374,4.263), (51.798,6.065), (51.874,6.133), (53.752,6.349), (53.526,4.370)]

# Create a Shapely Polygon object
polygon = Polygon(polygon_coords)

# Function to check if a point (latitude, longitude) is within the polygon
def is_within_polygon(lat, lon):
    point = Point(lat, lon)
    return polygon.contains(point)

# Filter DataFrame based on the polygon
df = data_normalized[data_normalized.apply(lambda row: is_within_polygon(row['navigation.location.lat'], row['navigation.location.long']), axis=1)]

# Drop duplicates
df = df.drop_duplicates(subset=['vessel.imo', 'navigation.time'], keep='last')

# Ekkes spaties weghalen
df['navigation.status'] = df['navigation.status'].str.strip()

df


# In[68]:


# df['vessel.callsign'].unique()
# df['device.mmsi'].unique()
# df['vessel.imo'].unique()

# Deze is waarschijnlijk het meest betrouwbaar gezien schepen echt hun eigen naam hebben. Niemand gebruikt dubbele namen.
df['vessel.name'].unique()
df['vessel.name'].value_counts()


# In[69]:


vessel_name_counts = df['vessel.name'].value_counts()
df = df[df['vessel.name'].isin(vessel_name_counts.index[vessel_name_counts > 5])]

df


# In[70]:


df['vessel.type'].value_counts()


# In[71]:


# all_locations = plt.scatter(df['navigation.location.long'], df['navigation.location.lat'])


# # In[72]:


# df_test1 = df[(df['vessel.type'] == 'cargo')]
# locations = plt.scatter(df_test1['navigation.location.long'], df_test1['navigation.location.lat'])


# # In[73]:


# df_test2 = df[(df['vessel.type'] == 'other')]
# locations = plt.scatter(df_test2['navigation.location.long'], df_test2['navigation.location.lat'])


# # In[74]:


# df_test3 = df[(df['vessel.type'] == 'dredging-underwater-ops')]
# locations = plt.scatter(df_test3['navigation.location.long'], df_test3['navigation.location.lat'])


# # In[75]:


# df_test4 = df[(df['vessel.type'] == 'tanker')]
# locations = plt.scatter(df_test4['navigation.location.long'], df_test4['navigation.location.lat'])


# # In[76]:


# # Create a folium map centered at the mean latitude and longitude
# map_center = [df['navigation.location.lat'].mean(), df['navigation.location.long'].mean()]
# m = folium.Map(location=map_center, zoom_start=10)

# marker_cluster = MarkerCluster().add_to(m)

# for index, row in df.iterrows():
#     folium.Marker(location=[row['navigation.location.lat'], row['navigation.location.long']]).add_to(marker_cluster)

# scatter_html = mplleaflet.fig_to_html(plt.gcf())

# # Create a folium iframe to embed the scatterplot HTML
# scatter_frame = folium.IFrame(html=scatter_html, width=500, height=300)
# scatter_popup = folium.Popup(scatter_frame, max_width=500)

# # Add the scatterplot as a popup to the folium map
# folium.Marker(location=map_center, popup=scatter_popup).add_to(m)
    
# # m.save('filtered_data.html')
# # m


# In[91]:


# Functie om string naar datetime object om te zetten
def convert_to_datetime(datetime_str):
    return pd.to_datetime(datetime_str)

# Nieuwe kolommen toevoegen
df['navigation.time'] = df['navigation.time'].apply(convert_to_datetime)
df['date'] = df['navigation.time'].dt.date
df['time'] = df['navigation.time'].dt.time

# subset = df.iloc[19719:19730]
# print(subset)


# In[111]:


df = df.sort_values(by=['vessel.name', 'group', 'navigation.time'])

# Get the rows where 'navigation.status' and 'vessel.name' changes
df['status_change'] = (df['navigation.status'] != df['navigation.status'].shift(1)) | (df['vessel.name'] != df['vessel.name'].shift(1))

# Increment group number only when status changes
df['group'] = (df['status_change'] == True).cumsum()



# Group by 'vessel.name', 'group', and 'navigation.status' and aggregate start and end time
result = df.groupby(['vessel.name', 'group', 'navigation.status', 'navigation.draught']).agg(start_time=('time', 'first'), end_time=('time', 'last'), start_date=('date', 'first'), end_date=('date', 'last'))

# Format the output
for index, row in result.iterrows():
    print(f"{index[0]}: {row['start_date'].isoformat()} {row['start_time'].isoformat()} - {row['end_date'].isoformat()} {row['end_time'].isoformat()} {index[2]}")

result


# In[138]:


# # Calculate time differences between consecutive records
df['time_diff'] = df['navigation.time'].diff()

# Handle cases where navigation.status or vessel.name changes (i.e., start of a new group)
df['time_diff'] = df['time_diff'].where(df['status_change'] == False, pd.NaT)

# Forward fill NaN values to propagate the time difference across the entire group
df['time_diff'] = df.groupby(['vessel.name', 'group'])['time_diff'].ffill()

# Group by 'vessel.name', 'group', and 'navigation.status' and aggregate total time
final = df.groupby(['vessel.name', 'group', 'navigation.status', 'navigation.draught', 'vessel.type']).agg(total_time=('time_diff', 'sum'),
                                                                       start_time=('time', 'first'),
                                                                       end_time=('time', 'last'),
                                                                       start_date=('date', 'first'),
                                                                       end_date=('date', 'last')).reset_index()

# Format the output
for index, row in final.iterrows():
    total_time = row['total_time']
    total_minutes = total_time.total_seconds() / 60  # Convert total time to minutes
    print(f"{row['vessel.name']}: {row['start_date'].isoformat()} {row['start_time'].isoformat()} - {row['end_date'].isoformat()} {row['end_time'].isoformat()} {row['navigation.status']} - Total Time: {total_minutes:.2f} minutes")

final['total_time_minutes'] = final['total_time'].dt.total_seconds() / 60
final.drop(columns=['total_time'], inplace=True)

final




# In[121]:


final_df = final[final['navigation.status'] == 'moored']
final_df


# In[136]:


moored_time = final_df[['total_time_minutes', 'navigation.draught']]
moored_array = moored_time.values.tolist()

print(moored_array)
print(len(moored_array))

vessel_type = final_df[['vessel.type']]
vessel_array = vessel_type.values.tolist()

print(vessel_array)
print(len(vessel_array))


# In[137]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data into a training set and test set
# x = moored_array en y = vessel_array
moored_array_train, moored_array_test, vessel_array_train, vessel_array_test = train_test_split(moored_array, vessel_array, test_size=0.3, random_state=42)

# Train a logistic regression model using the training set
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty=None, max_iter=50)
model.fit(moored_array_train, vessel_array_train)

# predict results using the test set
vessel_array_prediction = model.predict(moored_array_test)

# Calculate accuracy
accuracy = accuracy_score(vessel_array_test, vessel_array_prediction)
print("Accuracy using Softmax Regression:", accuracy)


# Calculate the confusion matrix
conf_matrix = confusion_matrix(vessel_array_test, vessel_array_prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted vessel type')
plt.ylabel('True vessel type')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
class_report = classification_report(vessel_array_test, vessel_array_prediction)
print("Classification Report:\n", class_report)


# In[126]:


# print(df.to_string(index=False))

