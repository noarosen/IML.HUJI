from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
all_cols_mean = []  # global for preprocess


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
        preprocess data only on train set
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Design matrix of regression problem

        y : array-like of shape (n_samples, )
            Response vector corresponding given samples
        Returns
        -------
        Post-processed design matrix and response vector as a DataFrame
        """
    y = pd.to_numeric(y, errors='coerce')  # convert y to numeric
    X["price"] = y
    X = X.dropna().drop_duplicates()  # remove rows
    for col_to_int in ["view", "condition", "grade", "zipcode", "decade_built"]:
        X.loc[:, col_to_int] = X.loc[:, col_to_int].astype(int)
    # drop non-valid values - not in range/ do not make sense
    for col_pos in ["bedrooms", "bathrooms", "floors", "sqft_basement",
                    "yr_renovated", "zipcode"]:
        X = X[X[col_pos] >= 0]
    for col_pos in ["sqft_living", "sqft_lot", "sqft_above", "yr_built",
                    "price"]:
        X = X[X[col_pos] > 0]
    X = X[X["bedrooms"] < 15]
    X = X[X["bathrooms"] < 10]
    X = X[X["floors"] < 4]
    X = X[X["waterfront"].isin([0, 1])]
    X = X[X["view"].isin(range(5))]
    X = X[X["condition"].isin(range(1, 6))]
    X = X[X["grade"].isin(range(1, 15))]
    # dummies
    X = pd.get_dummies(X, prefix='zipcode_dummies', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='decade_built_dummies',
                       columns=['decade_built'])
    X = X.drop("yr_built", axis=1)
    # calculate all means
    for col in ["view", "condition", "grade", "bedrooms",
                "bathrooms", "floors", "sqft_basement", "yr_renovated",
                "sqft_living", "sqft_lot", "sqft_above",
                "waterfront"]:
        all_cols_mean.append(X[col].mean())
    all_cols_mean.append(0)  # for zipcode
    return X


def preprocess_test(X: pd.DataFrame):
    """
        preprocess data only on test set
        Parameter
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Design matrix of regression problem
        Returns
        -------
        Post-processed design matrix X
        """
    # replace NA values
    i = 0
    for col_replace_na in ["view", "condition", "grade", "bedrooms",
                "bathrooms", "floors", "sqft_basement", "yr_renovated",
                "sqft_living", "sqft_lot", "sqft_above",
                "waterfront", "zipcode"]:
        mean_value = all_cols_mean[i]
        # replace na and infinite values with the mean value
        X[col_replace_na] = X[col_replace_na].replace([np.nan, np.inf, -np.inf],
                                                      mean_value)
        i += 1
    for col_to_int in ["view", "condition", "grade", "zipcode"]:
        X.loc[:, col_to_int] = X.loc[:, col_to_int].astype(int)
    # chang to valid values in range/ that make sense
    for col_pos in ["bedrooms", "bathrooms", "floors", "sqft_basement",
                    "yr_renovated"]:
        X.loc[X[col_pos] < 0, col_pos] = 0
    for col_pos in ["sqft_living", "sqft_lot", "sqft_above", "yr_built"]:
        X.loc[X[col_pos] < 0, col_pos] = 1
    # chang to valid values in range/ that make sense
    X.loc[X["bedrooms"] > 15, "bedrooms"] = 15
    X.loc[X["bathrooms"] > 10, "bathrooms"] = 10
    X.loc[X["floors"] > 4, "floors"] = 4
    X.loc[X["waterfront"] < 0, "waterfront"] = 0
    X.loc[X["waterfront"] > 1, "waterfront"] = 1
    X.loc[X["view"] < 0, "view"] = 0
    X.loc[X["view"] > 4, "view"] = 4
    X.loc[X["condition"] < 1, "condition"] = 1
    X.loc[X["condition"] > 5, "condition"] = 5
    X.loc[X["grade"] < 1, "grade"] = 1
    X.loc[X["grade"] > 13, "grade"] = 13
    # dummies
    X["zipcode"] = X["zipcode"].astype(int)
    X = pd.get_dummies(X, prefix='zipcode_dummies', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='decade_built_dummies',
                       columns=['decade_built'])
    X = X.drop("yr_built", axis=1)
    return X


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # for train and test
    X = X.drop(["id", "date", "long", "lat", "sqft_lot15", "sqft_living15"],
               axis=1)  # remove irrelevant columns
    # new feature
    X["high_grade"] = np.where(X["grade"] >= 11, 1, 0)  # above 11 -> yes (1)

    X["decade_built"] = ((X["yr_built"] // 10) * 10)
    for col_numeric in ["view", "condition", "grade", "bedrooms",
                           "bathrooms", "floors", "sqft_basement",
                           "yr_renovated", "yr_built", "sqft_living",
                           "sqft_lot", "sqft_above", "waterfront", "zipcode"]:
        X[col_numeric] = pd.to_numeric(X[col_numeric], errors='coerce')
    X.replace('nan', np.nan)

    if np.any(y):  # train set only
        pros_X = preprocess_train(X, y)
        return pros_X.loc[:, pros_X.columns != "price"], pros_X.loc[:, "price"]
    else:  # for test set only
        pros_X = preprocess_test(X)
        return pros_X, None  # only X


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # remove dummies columns
    cols_to_drop = [col for col in X.columns if
                    'zipcode_dummies' in col or 'decade_built_dummies' in col]
    X = X.drop(cols_to_drop, axis=1)
    for feature in X.columns:
        curr_X = X[feature]
        cov = np.cov(curr_X, y)[0, 1]
        std = (np.std(curr_X) * np.std(y))
        rho = cov / std
        title = f"Correlation between {feature} and RHO: {rho}"
        plot = px.scatter(pd.DataFrame({'x': curr_X, 'y': y}), x="x", y="y",
                          trendline="ols", trendline_color_override="black",
                          color_discrete_sequence=["#ADD8E6"],
                          title=title,
                          labels={"x": f"{feature}", "y": "response"})
        plot.write_image(f"{output_path}/correlation_{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    X_data = df.loc[:, df.columns != "price"]  # only data without prices
    y_price = df.loc[:, "price"]
    train_X, train_y, test_X, test_y = split_train_test(X_data, y_price)

    # Question 2 - Preprocessing of housing prices dataset
    processed_train_X, processed_train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(processed_train_X, processed_train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    processed_test_X, processed_test_y = preprocess_data(test_X)
    processed_test_X = processed_test_X.reindex(
        columns=processed_train_X.columns, fill_value=0)
    if processed_test_y is None:
        processed_test_y = test_y  # no process needed
    all_p = list(range(10, 101))
    per_num = len(all_p)  # number of percentages
    test_results = np.zeros((per_num, 10))
    for i, p in enumerate(all_p):
        for j in range(test_results.shape[1]):
            X = processed_train_X.sample(frac=p / 100.0)
            y = processed_train_y.loc[X.index]
            test_results[i, j] = LinearRegression(include_intercept=True) \
                .fit(X, y).loss(processed_test_X, processed_test_y)

    mean = test_results.mean(axis=1)
    std = test_results.std(axis=1)
    top = mean - (std * 2)
    bottom = mean + (std * 2)
    title = "MSE as Function Of Training Set Size"
    title_x = "Percentage of training set"
    title_y = "MSE on Test Set"
    plot = go.Figure([go.Scatter(x=all_p, y=top, fill=None, mode="lines",
                                 line=dict(color="#ADD8E6")),
                      go.Scatter(x=all_p, y=bottom, mode="lines",
                                 line=dict(color="#ADD8E6")),
                      go.Scatter(x=all_p, y=mean, mode="lines",
                                 marker=dict(color="black"))],
                     layout=go.Layout(title=title, xaxis=dict(title=title_x),
                                      yaxis=dict(title=title_y),
                                      showlegend=False))
    plot.write_image("mse_as_trainsize.png")
