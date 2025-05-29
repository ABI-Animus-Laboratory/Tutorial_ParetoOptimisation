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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    
    #    Dataset generation
    noise_std = 0.3
    number_measurements = 200
    X_1d = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    Y_1d = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    X, Y = np.meshgrid(X_1d, Y_1d)
    f = np.squeeze(X * np.sin(X)* np.cos(Y))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, f, cmap=cm.viridis)
    plt.title(r"f(x)")
    # plt.show()
    
    rng = np.random.RandomState(1)
    training_indices_x = rng.choice(np.arange(f.shape[0]), size=number_measurements, replace=True)
    training_indices_y = rng.choice(np.arange(f.shape[1]), size=number_measurements, replace=True)
    X_train, f_train = np.stack([np.reshape(X[training_indices_x,training_indices_y],(-1,)),np.reshape(Y[training_indices_x,training_indices_y],(-1,))],axis=1), np.reshape(f[training_indices_x,training_indices_y],(-1,))
    f_train_noisy = f_train + rng.normal(loc=0.0, scale=noise_std, size=f_train.shape)

    #    Assembling regressor
    
    #    Choosing kernel - i.e. covariance between values of x
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))    
    # kernel = Matern(nu=2.5)
    # kernel = RationalQuadratic()
    
    #    Need alpha to have noise for numerical stability. Ensure that alpha is 
    #    orders of magnitude smaller than measurements uncertainty to avoid affecting your fitting!
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, f_train_noisy)
    mean_prediction, std_prediction = gaussian_process.predict(np.stack([np.reshape(X,(-1,)),np.reshape(Y,(-1,))],axis=1), return_std=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, np.reshape(mean_prediction,(f.shape)), cmap=cm.viridis)
    ax.scatter(X[training_indices_x,training_indices_y], Y[training_indices_x,training_indices_y], f[training_indices_x,training_indices_y], c='red', marker='o', s=25)
    plt.title(r"s(x)")
    plt.show()
    