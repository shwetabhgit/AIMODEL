# sales_prediction.py using Linear Regression
import pandas as pd  # For creating and manipulating dataframes
from sklearn.metrics import mean_squared_error  # For evaluating the model's performance
import matplotlib.pyplot as plt  # For visualizing data and results
import numpy as np  # For handling numerical operations and arrays
from sklearn.linear_model import LinearRegression  # For building the linear regression model
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets


# Columns: TV Budget ($), Radio Budget ($), Newspaper Budget ($), Sales (units)
data = np.array([
    [230.1, 37.8, 69.2, 22.1],
    [44.5, 39.3, 45.1, 10.4],
    [17.2, 45.9, 69.3, 9.3],
    [151.5, 41.3, 58.5, 18.5],
    [180.8, 10.8, 58.4, 12.9],
    [8.7, 48.9, 75.0, 7.2],
    [57.5, 32.8, 23.5, 11.8],
    [120.2, 19.6, 11.6, 13.2],
    [144.1, 16.0, 40.3, 15.6],
    [111.6, 12.6, 37.9, 12.2]
])

# Splitting the data into features (X) and target (y)
# Features are the independent variables (TV, Radio, and Newspaper budgets)
# Target is the dependent variable (Sales)
X = data[:, 0:3]  # Features: TV Budget, Radio Budget, Newspaper Budget
y = data[:, 3]    # Target: Sales

# Step 2: Splitting the data into training and testing sets
# Training data is used to train the model, while testing data is used to evaluate its performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# test_size=0.3 means 30% of the data will be used for testing
# random_state=42 ensures reproducibility of the split

# Step 3: Creating and training the linear regression model
model = LinearRegression()  # Initialize the linear regression model
model.fit(X_train, y_train)  # Train the model using the training data

# Step 4: Making predictions on the test set
# The model uses the learned coefficients to predict sales based on the test features
y_pred = model.predict(X_test)

# Step 5: Evaluating the model's performance using Mean Squared Error (MSE)
# MSE measures the average squared difference between the predicted and actual values
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)  # Print the MSE to assess model accuracy

# Display the model coefficients and intercept
# Coefficients represent the weights assigned to each feature
# Intercept is the constant term in the linear equation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 6: Visualizing the training results
# Compare actual sales values to predicted sales values for the training set
y_train_pred = model.predict(X_train)  # Predict sales for the training set
plt.figure(figsize=(10, 5))  # Set the figure size
plt.scatter(y_train, y_train_pred, color='blue', label='Training Data')  # Scatter plot of actual vs predicted values
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Perfect Fit')  # Line representing perfect predictions
plt.title('Training Data: Actual vs Predicted')  # Title of the plot
plt.xlabel('Actual Sales (units)')  # Label for x-axis
plt.ylabel('Predicted Sales (units)')  # Label for y-axis
plt.legend()  # Add legend to the plot
plt.show()  # Display the plot

# Step 7: Visualizing the test results
# Compare actual sales values to predicted sales values for the test set
plt.figure(figsize=(10, 5))  # Set the figure size
plt.scatter(y_test, y_pred, color='green', label='Test Data')  # Scatter plot of actual vs predicted values
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')  # Line representing perfect predictions
plt.title('Test Data: Actual vs Predicted')  # Title of the plot
plt.xlabel('Actual Sales (units)')  # Label for x-axis
plt.ylabel('Predicted Sales (units)')  # Label for y-axis
plt.legend()  # Add legend to the plot
plt.show()  # Display the plot

# Step 8: Predicting sales based on user input
# Allow the user to input advertising budgets for TV, Radio, and Newspaper
tv_budget = float(input("Enter TV Budget ($): "))  # Get TV budget from user
radio_budget = float(input("Enter Radio Budget ($): "))  # Get Radio budget from user
newspaper_budget = float(input("Enter Newspaper Budget ($): "))  # Get Newspaper budget from user

# Predict sales based on user input
# Convert user input into a 2D numpy array to ensure compatibility with the model
user_input = np.array([[tv_budget, radio_budget, newspaper_budget]])
predicted_sales = model.predict(user_input)  # Predict sales using the trained model
print("Predicted Sales (units):", predicted_sales[0])  # Display the predicted sales value