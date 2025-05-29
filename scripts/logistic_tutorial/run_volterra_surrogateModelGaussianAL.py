'''
Created on 25/05/2025

@summary: Example of a surrogate model for the Lotka-Volterra model using Gaussian processes.

@author: Gonzalo D. Maso Talou
'''

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from modAL.models import ActiveLearner

import matplotlib.pyplot as plt

from models.LotkaVolterraModel import LotkaVolterra

def dataset_generation(num_samples,num_timesteps,beta_range,delta_range,p_0=5000,h_0=10,alpha=1.0,gamma=0.1,T=28,dt=0.5):
    
    betas = np.random.uniform(beta_range[0], beta_range[1], size=(num_samples))
    deltas = np.random.uniform(delta_range[0], delta_range[1], size=(num_samples))
    
    inputs = np.stack([betas,deltas]).transpose()
    # outputs = np.zeros((num_samples,2*num_timesteps))
    outputs = np.zeros((num_samples,num_timesteps))
    
    model = LotkaVolterra()
    for idx_sample in range(0,num_samples):
        t,prey,hunter = model.predict(p_0,h_0,alpha,betas[idx_sample],gamma,deltas[idx_sample],T,dt)
        # outputs[idx_sample,:] = np.concat([prey[0:num_timesteps],hunter[0:num_timesteps]])
        outputs[idx_sample,:] = hunter[0:num_timesteps]
        
    return inputs, outputs

def top_n_argmax(array, n):
    idx = np.argpartition(array, -n)[-n:]
    return idx[np.argsort(array[idx])][::-1]
      
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = top_n_argmax(np.mean(std,axis=1),n_batch)
    return query_idx, X[query_idx]

if __name__ == '__main__':
    
    #    Initial conditions
    p_0=5000
    h_0=10
    T=28
    dt=1.0
    num_timesteps=14
    initial = 500
    
    #    Model parameters - target GT
    alpha = 1.0
    beta = 0.05
    gamma = 0.1
    delta = 0.0001
    
    #    Model creation and prediction
    model = LotkaVolterra()
    t,prey_GT,hunter_GT = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
    
    #    Generate dataset for surrogate
    X, Y = dataset_generation(num_samples=5000,num_timesteps=num_timesteps,beta_range=[0.01,0.10],delta_range=[0.00001,0.0001],p_0=p_0,h_0=h_0,alpha=alpha,gamma=gamma,T=T,dt=dt)
    
    #    Assembling regressor   
    kernel = RBF()    
    # kernel = Matern(nu=1.5)
    # kernel = RationalQuadratic()
    
    regressor = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel),
                              query_strategy=GP_regression_std,
                              X_training=X[0:initial], y_training=Y[0:initial])

    
    X_gt = [[beta,delta]]
    mean_prediction, std_prediction = regressor.predict(X_gt, return_std=True)

    # plt.plot(t[0:num_timesteps], prey_GT[0:num_timesteps], label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.plot(t[0:num_timesteps], hunter_GT[0:num_timesteps], label=r"$h_{GT}(t)$", linestyle="dotted")
    plt.plot(t[0:num_timesteps], mean_prediction[0,0:num_timesteps], label="Mean prediction")
    plt.fill_between(
        t[0:num_timesteps].ravel(),
        mean_prediction[0,0:num_timesteps] - 1.96 * std_prediction[0,0:num_timesteps],
        mean_prediction[0,0:num_timesteps] + 1.96 * std_prediction[0,0:num_timesteps],
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("r$t$")
    plt.ylabel("r$h(t)$")
    _ = plt.title("Initial Gaussian process")
    
    plt.show(block=False)
    plt.pause(0.1)
    
    n_batch = 5
    n_queries = 100
    for idx in range(n_queries):
        print("Batch {}".format(idx))
        
        query_idx, query_instance = regressor.query(X)
        regressor.teach(X[query_idx], Y[query_idx])
        
        X_gt = [[beta,delta]]
        mean_prediction, std_prediction = regressor.predict(X_gt, return_std=True)
    
        # plt.plot(t[0:num_timesteps], prey_GT[0:num_timesteps], label=r"$f(x) = x \sin(x)$", linestyle="dotted")
        # plt.plot(t[0:num_timesteps], hunter_GT[0:num_timesteps], label=r"$h_{GT}(t)$", linestyle="dotted")
        plt.plot(t[0:num_timesteps], mean_prediction[0,0:num_timesteps], label="Mean prediction")
        plt.fill_between(
            t[0:num_timesteps].ravel(),
            mean_prediction[0,0:num_timesteps] - 1.96 * std_prediction[0,0:num_timesteps],
            mean_prediction[0,0:num_timesteps] + 1.96 * std_prediction[0,0:num_timesteps],
            color="tab:orange",
            alpha=0.5,
            label=r"95% confidence interval",
        )
        # plt.legend()
        # plt.xlabel("r$t$")
        # plt.ylabel("r$h(t)$")
        # _ = plt.title("Gaussian process - Batch {}".format(idx))
        plt.show(block=False)
        plt.pause(0.1)
        
        print("Total samples:{}".format(initial+n_batch*(idx+1)))

