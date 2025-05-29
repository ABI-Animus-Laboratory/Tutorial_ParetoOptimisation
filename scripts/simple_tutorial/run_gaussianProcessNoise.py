'''
Created on 25/05/2025

@author: Gonzalo D. Maso Talou
'''

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #    Dataset generation
    noise_std = 0.3
    number_measurements = 15
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    f = np.squeeze(X * np.sin(X))
    
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(f.size), size=number_measurements, replace=False)
    X_train, y_train = X[training_indices], f[training_indices]
    y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

    #    Assembling regressor
    
    #    Choosing kernel - i.e. covariance between values of x
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))    
    # kernel = Matern(nu=2.5)
    # kernel = RationalQuadratic()
    
    #    Need alpha to have noise for numerical stability. Ensure that alpha is 
    #    orders of magnitude smaller than measurements uncertainty to avoid affecting your fitting!
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train_noisy)
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    
    
    #    Plotting results
    plt.plot(X, f, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.errorbar(
        X_train,
        y_train_noisy,
        noise_std,
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on a noisy dataset")
    
    plt.show()