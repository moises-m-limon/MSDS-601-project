from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_smooth,
    geom_hline,
    geom_segment,
    ggtitle,
)


def ss_decomp(X, y, include_quadratic=False):

    # Fit a linear regression model

    if include_quadratic:
        X = np.column_stack((X, X**2)) 


    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate the mean of y
    y_mean = np.mean(y)

    # calculate SS quantities
    SST = np.sum((y - y_mean) ** 2).round(4)
    SSR = np.sum((y_pred - y_mean) ** 2).round(4)
    SSE = np.sum((y - y_pred) ** 2).round(4)

    if include_quadratic:
        X_plot = X[:, 0]  # The original X (without X^2) for plotting
    else:
        X_plot = X.flatten()

    # make a ggplot
    df = pd.DataFrame({"X": X_plot, "y": y, "predicted": y_pred, "y_mean": y_mean})

    gg1 = (
        ggplot(df, aes(x="X", y="y"))
        + geom_point()
        + geom_smooth(method="lm", formula="y ~ x + I(x**2)" if include_quadratic else "y ~ x", se=False)
        + geom_hline(yintercept=y_mean, linetype="dashed")
        + geom_segment(
            aes(xend="X", y="predicted", yend="y_mean"), color="blue", linetype="dashed"
        )  # SSR components
        + geom_segment(
            aes(xend="X", yend="predicted"), color="red", linetype="dashed"
        )  # SSE components
        + ggtitle(f"R^2 is {np.round(1-SSE/SST, 4)}")
    )

    return gg1
