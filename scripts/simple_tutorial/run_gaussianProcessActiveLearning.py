'''
Created on 25/05/2025

@summary: Example of active learning for Gaussian Process Regression

@author: Gonzalo D. Maso Talou
'''

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic

from modAL.models import ActiveLearner

import matplotlib.pyplot as plt

def plot_results(X,f,X_train,y_train,noise_std,mean_prediction,std_prediction):
    plt.plot(X, f, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.errorbar(
        X_train,
        y_train,
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

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

if __name__ == '__main__':
    
    #    Dataset generation
    noise_std = 0.0
    number_samples = 3
    number_AL_samples = 15

    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    f = np.squeeze(X * np.sin(X))
    
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(f.size), size=number_samples, replace=False)
    X_train, y_train = X[training_indices], f[training_indices]

    #    Assembling regressor
    
    #    Choosing kernel - i.e. covariance between values of x
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e2))    
    # kernel = Matern(nu=2.5)
    # kernel = RationalQuadratic()
    
    #    Need alpha to have noise for numerical stability. Ensure that alpha is 
    #    orders of magnitude smaller than measurements uncertainty to avoid affecting your fitting!
    regressor = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel),
                                query_strategy=GP_regression_std,
                                X_training=X_train, y_training=y_train)
    mean_prediction, std_prediction = regressor.predict(X, return_std=True)

    #    Plotting results
    plot_results(X,f,X_train,y_train,noise_std,mean_prediction,std_prediction)
    
    for idx in range(number_AL_samples):
        query_idx, query_instance = regressor.query(X)
        regressor.teach([X[query_idx]], [f[query_idx]])
        
        X_train = np.concatenate((X_train,X[query_idx].reshape(1,1)))
        y_train = np.concatenate((y_train,f[query_idx].reshape(-1)))
        mean_prediction, std_prediction = regressor.predict(X, return_std=True)
        #    Plotting results
        plot_results(X,f,X_train,y_train,noise_std,mean_prediction,std_prediction)
