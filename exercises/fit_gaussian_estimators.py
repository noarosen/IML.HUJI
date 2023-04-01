from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    x = np.random.normal(mu, var, 1000)

    fitted = UnivariateGaussian().fit(x)
    print((np.round(fitted.mu_, 3), np.round(fitted.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    mu_plot = [np.abs(mu - UnivariateGaussian().fit(x[:n]).mu_)
               for n in np.arange(var, len(x) / mu, dtype=np.int32) * mu]

    x_size = len(mu_plot)
    title1 = "Expectation Value Error as Function of Sample Size"
    go.Figure(go.Scatter(x=list(range(x_size)), y=mu_plot, mode="markers",
                         marker=dict(color="cornflowerblue")),
              layout=dict(template=pio.templates.default, title=title1,
                          xaxis_title=r"$\text{Sample size }$",
                          yaxis_title=r"$\text{Sample mean estimator}$")) \
        .write_image("PlotQ2.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = np.c_[x, fitted.pdf(x)]  # adding feature of pdf to X
    title2 = "PDF Plot - Fitted Model"
    go.Figure(go.Scatter(x=pdf[:, 0], y=pdf[:, 1], mode="markers",
                         marker=dict(color="cornflowerblue")),
              layout=dict(template=pio.templates.default, title=title2,
                          xaxis_title=r"$\text{Data value of fitted model}$",
                          yaxis_title=r"$\text{PDF value}$")) \
        .write_image("PlotQ3PDF.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    x = np.random.multivariate_normal(mu, cov, 1000)
    fitted = MultivariateGaussian().fit(x)
    print(np.round(fitted.mu_, 3))
    print(np.round(fitted.cov_, 3))

    # Question 5 - Likelihood evaluation
    matrix = np.zeros((200, 200))
    values = np.linspace(-10, 10, matrix.shape[0])  # log likelihood
    for i, f1 in enumerate(values):
        for j, f3 in enumerate(values):
            ll_ij = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, x)
            matrix[i, j] = np.round(ll_ij, 3)

    title3 = "Log-Likelihood as Function of Expectation of Features 1,3"
    go.Figure(go.Heatmap(x=values, y=values, z=matrix),
              layout=dict(template=pio.templates.default, title=title3,
                          xaxis_title=r"$\text{Mu by feature 3}$",
                          yaxis_title=r"$\text{Mu by feature 1}$")) \
        .write_image("PlotQ5HEATMAP.png")

    # Question 6 - Maximum likelihood
    max_index = list(np.unravel_index(matrix.argmax(), matrix.shape))
    max_ll = values[max_index]
    print("The model that achieved tha maximum log-likelihood is: ", np.round(max_ll, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
