'''
Created on 25/05/2025

@summary: Example of a surrogate model for the Lotka-Volterra model using Gaussian processes.

@author: Gonzalo D. Maso Talou
'''

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

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
    

if __name__ == '__main__':
    
    #    Initial conditions
    p_0=5000
    h_0=10
    T=28
    dt=1.0
    num_timesteps=8
    
    #    Model parameters - target GT
    alpha = 1.0
    beta = 0.05
    gamma = 0.1
    delta = 0.0001
    
    #    Model creation and prediction
    model = LotkaVolterra()
    t,prey_GT,hunter_GT = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
    
    #    Generate dataset for surrogate
    x_train, y_train = dataset_generation(num_samples=500,num_timesteps=num_timesteps,beta_range=[0.01,0.10],delta_range=[0.00001,0.0001],p_0=p_0,h_0=h_0,alpha=alpha,gamma=gamma,T=T,dt=dt)
    
    #    Assembling regressor   
    kernel = RBF()    
    # kernel = Matern(nu=1.5)
    # kernel = RationalQuadratic()
    
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9, n_targets=num_timesteps, normalize_y=True)
    gaussian_process.fit(x_train, y_train)
    X = [[beta,delta]]
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

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
    _ = plt.title("Gaussian process regression on hunter output")
    
    plt.show()
