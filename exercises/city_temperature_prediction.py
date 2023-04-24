import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    for col in df.columns:
        print(col, ":", df[col].dtype)
    df["DayOfYear"] = df["Date"].dt.dayofyear  # new col
    df = df.dropna().drop_duplicates()  # general
    df["Date"] = pd.to_datetime(df["Date"])  # date
    df = df[df.Temp > -10]  # temp
    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    il_df = df[df.Country == "Israel"]
    title2_1 = "Temp in Tel Aviv as a function of day of the year"
    plot2_1 = px.scatter(il_df, x="DayOfYear", y="Temp", color="Year", title=title2_1)
    plot2_1.update_xaxes(title="Day of Year")
    plot2_1.write_image("Q21_temp_tlv_days.png")

    title2_2 = "Standard deviation of daily temperatures by month"
    plot2_2 = px.bar(il_df.groupby(["Month"], as_index=False).agg(std=("Temp", "std")),
                   x="Month", y="std", title=title2_2, color_discrete_sequence=["#ADD8E6"])
    plot2_2.write_image("Q22_std_temp_tlv.png")

    # Question 3 - Exploring differences between countries
    title3 = "Mean temp per month in different countries"
    plot3 = px.line(df.groupby(["Country", "Month"], as_index=False).agg(std=("Temp", "std"), mean=("Temp", "mean")),
                    x="Month", y="mean", title=title3, color="Country", error_y="std")
    plot3.update_yaxes(title="Mean temp C")
    plot3.write_image("Q3_temp_months_by_country.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(il_df.DayOfYear, il_df.Temp)
    all_k = np.arange(1, 11)
    loss = np.zeros(all_k.shape, dtype=np.float64)
    for i, k in enumerate(all_k):
        model = PolynomialFitting(k=k).fit(train_X.values, train_y.values)
        loss[i] = np.round(model.loss(test_X.values, test_y.values), 2)

    loss_values = pd.DataFrame(dict(k=all_k, loss=loss))
    print(loss_values)
    title4 = "Test error for each k value"
    plot4 = px.bar(loss_values, x="k", y="loss", text="loss", title=title4,
                   color_discrete_sequence=["#ADD8E6"])
    plot4.write_image("Q4_loss_per_k_il.png")

    # Question 5 - Evaluating fitted model on different countries
    title5 = "Loss evaluation of fitted model on different countries"
    model = PolynomialFitting(k=5).fit(il_df.DayOfYear.values,
                                       il_df.Temp.values)
    plot5 = px.bar(pd.DataFrame([{"country": c, "loss": round(
        model.loss(df[df.Country == c].DayOfYear, df[df.Country == c].Temp), 2)}
                         for c in
                         ["Jordan", "South Africa", "The Netherlands"]]),
           x="country", y="loss", text="loss", color="country",
           title=title5)
    plot5.write_image("Q5_loss_of_countries.png")