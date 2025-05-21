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
    dt=0.25
    
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
    
    #    Calibrate model
    def F(theta,alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement):
        beta,delta = theta
        
        t,prey,hunter = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
        
        return np.square(np.subtract(prey,prey_measurement)).sum() + np.square(np.subtract(hunter,hunter_measurement)).sum()  
         
    res = minimize(F, [1E-2,1E-2], args=(alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,hunter_measurement), method='L-BFGS-B',
                   options={'disp': True, 'gtol':1E-8, 'maxls':50})

    theta_res = res.x
    print(theta_res)
    
    
    #    Compute profile likelihood
    samples_per_profile = 41
    min_factor = 0.1
    max_factor = 5
    
    #    Modified objective function for profile likelihood
    def F_likelihood(theta, alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement, param_idx, param_value):
        theta_full = np.insert(theta,param_idx,param_value[param_idx])
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
            theta_profile = theta_res
            theta_profile[param_idx] = current_value
            res = minimize(F_likelihood, theta_init, args=(alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,hunter_measurement,param_idx,theta_res), method='L-BFGS-B', options={'gtol':1E-6})
            profiles[param_idx,idx_sample] = res.fun
            # print(res)
    
    print(profiles)
    
    #    Print profiles
    plt.figure(figsize=(10, 5))
    for idx_profile in range(0,profiles.shape[0]):
        
        plt.subplot(1, profiles.shape[0], idx_profile+1)
        plt.plot(ranges[idx_profile,:], profiles[idx_profile,:], label='Prey')
        plt.xlabel('Range')
        plt.ylabel('Profile')
        plt.legend()

    plt.show()
        
        