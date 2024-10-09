from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_histogram, ggtitle


def display_resid_histogram(X, Y, name):
    X = np.array(X).reshape(-1, 1)  # Ensure X is in the correct shape for sklearn
    Y = np.array(Y)  # Convert Y to numpy array if needed

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = Y - y_pred

    #some binwidth calculations chat gave me
    q75, q25 = np.percentile(residuals, [75, 25])
    iqr = q75 - q25
    binwidth = 2 * iqr / np.cbrt(len(residuals)) 

    # Create a DataFrame for the histogram
    residuals_df = pd.DataFrame({
        'resid': residuals
    })

    # Create the ggplot for the residuals histogram
    plot = (
        ggplot(residuals_df, aes(x='resid'))
        + geom_histogram(binwidth=binwidth, fill="blue", color="black")  # Histogram of residuals
        + ggtitle(name)  # Title of the plot
    )

    return plot

