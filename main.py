import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
housing = fetch_california_housing(as_frame=True)
housing = housing.frame

# Define features (X) and target (y)
X = housing.iloc[:, :-1]
y = housing.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for regression
regression_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train the model
regression_pipeline.fit(X_train, y_train)

# Function to predict house value based on user input
def predict_house_value():
    print("Please enter the following values to predict the median house value:")
    MedInc = float(input("Median Income in 10,000s: "))   # Example input: 5.0 (represents $50,000)
    HouseAge = float(input("House Age (median): "))       # Example input: 30 (years)
    AveRooms = float(input("Average number of rooms: "))  # Example input: 5
    AveBedrms = float(input("Average number of bedrooms: "))  # Example input: 1
    Population = float(input("Population: "))             # Example input: 1000
    AveOccup = float(input("Average household size: "))   # Example input: 3
    Latitude = float(input("Latitude: "))                 # Example input: 34.0
    Longitude = float(input("Longitude: "))               # Example input: -118.0

    # Create a DataFrame from the input values
    user_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
                             columns=X.columns)

    # Predict house value using the trained model
    predicted_value = regression_pipeline.predict(user_data)

    print(f"The predicted median house value is: ${predicted_value[0] * 100000:.2f}")

# Call the function to predict house value based on user input
predict_house_value()

plt.figure(figsize=(8, 6))
plt.scatter(X['MedInc'], y, alpha=0.5)
plt.xlabel('Median Income in 10,000s')
plt.ylabel('Median House Value in 100,000s')
plt.title('Scatter Plot of Median Income vs. Median House Value')
plt.show()