import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Load the California Housing dataset
california = fetch_california_housing()
X = california.data[:, np.newaxis, 0]  # Only use the first feature for simplicity (MedInc)
y = california.target  # House values

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict house values using the test data
y_pred = model.predict(X_test)

# Step 5: Plot the results
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred, color='red', label="Regression Line")
plt.xlabel('Median Income (MedInc)')
plt.ylabel('House Value')
plt.title('Linear Regression on California Housing Dataset')
plt.legend()
plt.show()
