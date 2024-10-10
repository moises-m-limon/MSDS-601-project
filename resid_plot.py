from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_hline, geom_segment, ggtitle

# Define the function to display residuals using X and Y as inputs
def display_resid(X, Y, name):
    # Convert X to a 2D array for sklearn if it's not already
    X = np.array(X).reshape(-1, 1)  # Ensure X is in the correct shape for sklearn
    Y = np.array(Y)  # Convert Y to numpy array if needed

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = Y - y_pred

    # Create a DataFrame for plotting
    residuals_df = pd.DataFrame({
        'x': X.flatten(),  # Flatten X to ensure it's 1D for plotting
        'resid': residuals
    })

    # Create the ggplot for residuals
    plot = (
        ggplot(residuals_df, aes(x='y_pred', y='resid'))
        + geom_point(size=2, color="blue")  # Scatter plot of residuals
        + geom_hline(yintercept=0, color="green")  # Horizontal line at y=0
        + geom_segment(aes(x='y_pred', xend='y_pred', y=0, yend='resid'), color='gray', size=0.5)  # Vertical lines for residuals
        + ggtitle(name)  # Title
    )

    return plot