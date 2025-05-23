'''
Created on 21/05/2025

@summary: Example of single-objective optimisation.

@author: Gonzalo D. Maso Talou
'''

from models.LotkaVolterraModel import LotkaVolterra

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__ == '__main__':
    
    #    Initial conditions
    p_0=5000
    h_0=10
    T=14
    dt=0.05
    
    #    Ground truth model parameters
    alpha_gt = 1.0
    beta_gt = 5E-2
    gamma_gt = 0.25
    delta_gt = 1E-4
    
    #    Noise in the measurements
    sigma_prey=100.0
    sigma_hunter=3.0
    
    #    Measurements generation
    model = LotkaVolterra()
    t,prey, hunter, prey_measurement,hunter_measurement = model.generate_measurements(sigma_prey=sigma_prey,sigma_hunter=sigma_hunter,p_0=p_0,h_0=h_0,alpha=alpha_gt,beta=beta_gt,gamma=gamma_gt,delta=delta_gt,T=T,dt=dt)
    # model.plot_prey_hunter_measurements(t, prey, hunter, prey_measurement, hunter_measurement)
    theta_names = ['beta','delta']
    
    #    Calibrate model
    def F(theta,alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement):
        beta,delta = theta
        
        t,prey,hunter = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
        
        return np.square(np.subtract(prey,prey_measurement)).sum() + np.square(np.subtract(hunter,hunter_measurement)).sum()  
         
    res = minimize(F, [1E-2,1E-2], args=(alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,hunter_measurement), method='L-BFGS-B',
                   options={'disp': True, 'gtol':1E-8, 'maxls':50})

    theta_res = res.x
    best_fun = res.fun
    print(theta_res)

    # plot best fit result
    t,prey,hunter = model.predict(p_0,h_0,alpha_gt,theta_res[0],gamma_gt,theta_res[1],T,dt)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(t, prey, label='Prey')
    plt.plot(t, prey_measurement, 'o', label='Prey Measurement')
    plt.xlabel('Time')
    plt.ylabel('Prey Population')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(t, hunter, label='Hunter')
    plt.plot(t, hunter_measurement, 'o', label='Hunter Measurement')
    plt.xlabel('Time')
    plt.ylabel('Hunter Population')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(prey, hunter, label='Hunter vs Prey')
    plt.plot(prey_measurement, hunter_measurement, 'o', label='Hunter vs Prey Measurement')
    plt.xlabel('prey population')
    plt.ylabel('hunter population')
    plt.legend()
    plt.show()
    
    
    #    Compute profile likelihood
    samples_per_profile = 81
    min_factor = 0.1
    max_factor = 5
    
    #    Modified objective function for profile likelihood
    def F_likelihood(theta, alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement, param_idx, current_value):
        theta_full = np.insert(theta,param_idx,current_value)
        beta,delta = theta_full
        
        t,prey,hunter = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
        
        return np.square(np.subtract(prey,prey_measurement)).sum() + np.square(np.subtract(hunter,hunter_measurement)).sum()  
    
    #    Iterating over each variable and each sample of the profile
    profiles = np.zeros( (len(theta_res),samples_per_profile) )
    ranges = np.zeros( (len(theta_res),samples_per_profile) )
    for param_idx in range(0,len(theta_res)):
        theta_init = np.delete(theta_res,param_idx)
        
        param_value = theta_res[param_idx]
        min_range = min_factor * param_value 
        max_range = max_factor * param_value
        
        param_range = np.linspace(min_range,max_range, samples_per_profile)
        ranges[param_idx,:] = param_range
        for idx_sample, current_value in enumerate(param_range): 
            res = minimize(F_likelihood, theta_init, args=(alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,hunter_measurement,param_idx,current_value), method='L-BFGS-B', options={'gtol':1E-6})
            profiles[param_idx,idx_sample] = res.fun
            # print(res)
    
    print(profiles)
    
    #    Print profiles
    plt.figure(figsize=(10, 5))
    for idx_profile in range(0,profiles.shape[0]):
        
        plt.subplot(1, profiles.shape[0], idx_profile+1)
        plt.plot(ranges[idx_profile,:], profiles[idx_profile,:])
        plt.xlabel(theta_names[idx_profile])
        plt.ylabel('cost')
        plt.legend()

        # plot over with best fit
        plt.plot(theta_res[idx_profile], best_fun, 'rx', label='Best Fit')
    

    plt.show()
        
        