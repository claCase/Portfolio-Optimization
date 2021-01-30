import numpy as np
import matplotlib.pyplot as plt
import os
import get_data
from sklearn.datasets import make_spd_matrix
import pandas as pd
from pypfopt import efficient_frontier, expected_returns, risk_models, plotting, objective_functions
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
import json


def portfolio_optimization():
    optimal_path = "optimal_weights.txt"
    if not os.path.exists(optimal_path):
        data_path = os.path.join("tickers_data", "all_data.csv")
        if not os.path.exists(data_path):
            data = get_data.load_data()  # [["Ticker", "close"]].groupby(["Ticker"]).T
        else:
            data = pd.read_csv(data_path)

        data_series = pd.pivot_table(data, index="datetime", columns="Ticker", values="close")
        # print(data_series.head())
        mu = expected_returns.ema_historical_return(data_series)
        cov = risk_models.exp_cov(data_series)
        # plotting.plot_covariance(cov, plot_correlation=True)
        # print(mu, cov)
        ef = efficient_frontier.EfficientFrontier(mu, cov, weight_bounds=(0, 1))
        ef.add_objective(objective_functions.L2_reg, gamma=1)
        ef.max_sharpe(0.002)
        weights_portfolio = ef.weights

        # ef.max_sharpe(risk_free_rate=0.002)
        # ef.max_sharpe()
        dict_keys = data_series.columns.values.tolist()
        # print(dict_keys)

        weights = {}
        for key, value in zip(dict_keys, weights_portfolio):
            # print(f"{key} - {value}")
            weights[key] = value

        # print("SORTED WEIGHTS")
        sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))
        '''for key in sorted_weights.keys():
            print(f"{key} - {sorted_weights[key]}")
        '''
        cleaned_weights = {k: v for k, v in sorted_weights.items() if v > 10e-4}
        with open(optimal_path, "w") as file:
            file.write(json.dumps(cleaned_weights))
        # plt.pie(cleaned_weights.values(), labels=cleaned_weights.keys())
        # plt.show()
    else:
        with open(optimal_path, "r") as file:
            cleaned_weights = json.loads(file.read())

    return cleaned_weights


means = [15, 10, 30, 12]
covs = make_spd_matrix(len(means))
returns_data_dist = sp.stats.multivariate_normal(mean=means, cov=covs)
returns_data = returns_data_dist.rvs(size=10000)


def portfolio_variance(weights, cov, mu):
    weights = np.asarray(weights)
    pvs = np.dot(weights, cov).dot(weights.T).T
    return pvs


def portfolio_returns(weights, mus):
    return np.dot(weights, mus)


def visualize_return_variance(mus, cov, optimal_weights=None):
    from itertools import product
    x = np.linspace(0, 1, 100)
    w = np.asarray(np.meshgrid( * [x] * len(mus))).T.reshape(-1, len(mus))
    w = w / w.sum(axis=1)[:, np.newaxis]
    # print(w[:10])
    ret = portfolio_returns(w, mus)
    variances = []
    for weight in w:
        var = portfolio_variance(weight, cov, mus)
        variances.append(var)
    if optimal_weights is not None:
        optimal_weights = np.asarray(list(optimal_weights))
        optimal_weights_return = portfolio_returns(optimal_weights, mus)
        optimal_weights_variance = portfolio_variance(optimal_weights, cov, mus)

    print("returns: {} \n ret_shape: {}".format(ret[:4], ret.shape))
    print("variance: {} \n var_shape: {}".format(variances[:4], var.shape))
    plt.scatter(variances, ret)
    if optimal_weights is not None:
        plt.scatter(optimal_weights_variance, optimal_weights_return, color="red")
    plt.xlabel("Variance")
    plt.ylabel("Return")
    plt.show()


def objective(mus, cov, l):
    def function(weights):
        return -(portfolio_returns(weights, mus) + l * portfolio_variance(weights, cov, mus))

    return function


def constraints(weights):
    constr = np.sum(weights) - 1
    return constr


def mean_variance():
    weights = [1 / len(means)] * len(means)
    obj = objective(means, covs, 0.5)(weights)
    constr = constraints(weights)
    # print(obj)
    # print(constr)

    constr = {"type": "eq", "fun": constraints}
    sol = minimize(objective(means, covs, 1),
                   weights,
                   method="SLSQP",
                   # jac=True,
                   bounds=[(0, 1)] * len(means),
                   constraints=[constr])
    print(sol)

    # print(returns_data[:3])
