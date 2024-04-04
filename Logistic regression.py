from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Assuming X is a 2D array where each row represents a ship's mooring times and depth
X = [[6.4, 15.2], [8.9, 18.5], [11.2, 20.0], [4.7, 12.8], [10.1, 19.0],
     [2.5, 10.3], [7.8, 16.9], [9.3, 17.5], [3.6, 11.6], [5.2, 14.0],
     [13.4, 22.1], [18.2, 24.0], [20.5, 23.7], [14.8, 21.3], [19.7, 23.9],
     [12.1, 20.7], [17.3, 23.5], [16.6, 22.9], [15.0, 21.7], [21.9, 24.0],
     [22.3, 24.0], [23.7, 24.0], [1.9, 7.4], [2.2, 8.0], [3.3, 10.8],
     [5.5, 14.4], [7.1, 16.3], [8.8, 18.0], [9.9, 18.8], [11.0, 19.6],
     [13.2, 21.9], [14.5, 22.7], [16.8, 23.3], [17.7, 23.7], [18.9, 24.0],
     [20.2, 24.0], [21.3, 24.0], [22.6, 24.0], [24.0, 24.0], [2.8, 9.5],
     [3.9, 11.2], [4.2, 12.0], [6.6, 16.0], [8.0, 18.2], [9.1, 19.0],
     [10.5, 20.3], [11.8, 21.5], [12.9, 22.3], [15.3, 23.0], [16.1, 23.2],
     [18.5, 24.0], [19.4, 24.0], [20.7, 24.0], [21.8, 24.0], [23.0, 24.0],
     [1.7, 6.8], [2.4, 9.0], [3.1, 10.5], [4.8, 13.5], [5.7, 15.1],
     [6.9, 17.1], [7.6, 17.8], [8.3, 18.5], [9.6, 19.4], [11.3, 21.0],
     [13.5, 23.0], [14.2, 23.3], [15.7, 23.8], [16.4, 23.9], [17.9, 24.0],
     [19.2, 24.0], [20.6, 24.0], [22.1, 24.0], [23.6, 24.0], [1.5, 6.0],
     [2.1, 8.5], [4.1, 12.3], [5.4, 15.0], [6.2, 16.7], [7.4, 18.3],
     [8.7, 19.6], [10.0, 20.8], [12.3, 22.5], [13.7, 23.0], [14.9, 23.5],
     [16.3, 23.9], [17.5, 24.0], [18.7, 24.0], [20.1, 24.0], [21.5, 24.0],
     [22.7, 24.0], [24.0, 24.0], [1.3, 5.2], [3.0, 10.0], [4.4, 13.0],
     [5.9, 15.8], [6.7, 17.0], [7.9, 18.6], [9.2, 20.0], [10.4, 21.2],
     [11.7, 22.4], [13.1, 23.0], [14.3, 23.5], [15.6, 23.8], [16.9, 24.0]]

# Assuming y is a 1D array where each element represents the ship type (category)
y = ['Tanker', 'Container', 'Bulk Carrier', 'Fishing', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Container', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker',
     'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker', 'Tanker', 'Container', 'Bulk Carrier', 'Fishing', 'Container',
     'Tanker', 'Bulk Carrier', 'Fishing', 'Container', 'Tanker']

# Split the data into a training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model using the training set
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty=None)
model.fit(X_train, y_train)

# predict results using the test set
y_prediction = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy using Softmax Regression:", accuracy)


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted vessel type')
plt.ylabel('True vessel type')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
class_report = classification_report(y_test, y_prediction)
print("Classification Report:\n", class_report)

#test