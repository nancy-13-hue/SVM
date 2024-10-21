# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('titanic.csv')

# Display the first few rows to understand the data
print(data.head())

# Since 'Passenger' seems to be a unique identifier (likely irrelevant for prediction),
# we'll drop this column and use only 'Survived' as the target variable.
# Normally, SVM requires features, but with only two columns, we'll need more context
# to make meaningful predictions. For now, I'll assume 'Passenger' might contain some encoded values.

# Drop the 'Passenger' column
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM model (we can try different kernels like 'linear', 'rbf', etc.)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
