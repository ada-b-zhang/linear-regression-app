import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


advertising = pd.read_csv("advertising.csv")
# Extract TV, Radio, and Sales data for plotting
X = advertising[['TV', 'Radio']]  # Features (TV, Radio)
y = advertising['Sales']  # Target (Sales)

# Fit the linear regression model (degree = 1, no polynomial terms)
model = LinearRegression()
model.fit(X, y)

# Predict the corresponding Sales values using the linear model
y_pred = model.predict(X)

# Create a grid of TV and Radio values for the 3D plot
tv_range = np.linspace(X['TV'].min(), X['TV'].max(), 100)
radio_range = np.linspace(X['Radio'].min(), X['Radio'].max(), 100)
tv_grid, radio_grid = np.meshgrid(tv_range, radio_range)

# Create a new dataset for the grid points (flatten the grid)
X_grid = np.column_stack([tv_grid.ravel(), radio_grid.ravel()])

# Predict the Sales for the grid points using the fitted linear model
y_grid_pred = model.predict(X_grid)

# Reshape the predicted sales to match the shape of the TV and Radio grid
y_grid_pred = y_grid_pred.reshape(tv_grid.shape)

# Create the 3D plot
fig = go.Figure(data=[go.Surface(z=y_grid_pred, x=tv_grid, y=radio_grid, colorscale='Viridis', opacity=0.7)])

# Add the original data points as scatter points
fig.add_trace(go.Scatter3d(x=X['TV'], y=X['Radio'], z=y, mode='markers', marker=dict(size=4, color='orange'), name='Original Data'))

# Update layout for better visualization
fig.update_layout(
    title="Linear Regression with TV and Radio (Straight Plane)",
    scene=dict(
        xaxis_title="TV Advertising Spend ($)",
        yaxis_title="Radio Advertising Spend ($)",
        zaxis_title="Sales ($)"
    ),
    width=800,
    height=600
)

# Show the plot
fig.show()

# Calculate and print R² and adjusted R²
r2 = r2_score(y, y_pred)
n = len(y)  # Number of data points
p = X.shape[1]  # Number of predictors
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")

# Print the linear regression equation
intercept = model.intercept_
coefficients = model.coef_

equation = f"Sales = {intercept:.4f} + ({coefficients[0]:.4f} * TV) + ({coefficients[1]:.4f} * Radio)"
print("Linear Regression Equation (TV and Radio):")
print(equation)
