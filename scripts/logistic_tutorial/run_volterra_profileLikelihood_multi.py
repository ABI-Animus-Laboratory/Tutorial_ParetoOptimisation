'''
Created on 21/05/2025

@summary: Example of single-objective optimisation.

@author: Gonzalo D. Maso Talou, Finbar Argus
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
    dts = [0.1, 0.25, 0.5]
    
    #    Ground truth model parameters
    alpha_gt = 1.0
    beta_gt = 5E-2
    gamma_gt = 0.25
    delta_gt = 1E-4
    theta_gt = [beta_gt, delta_gt]  
        
    # Optimisation parameters
    methods = ['Nelder-Mead'] # 'Nelder-Mead' # 'L-BFGS-B' # 'Powell'
    
    #    Compute profile likelihood
    samples_per_profile = 41
    min_factor = 0.2
    max_factor = 2

    noise_mods = [1.0, 5.0]
    theta_0 = [1e-2, 1e-2]
        
    cost_subsets = [None, 0, 1]
    profiles = np.zeros( (len(noise_mods)*len(dts)*len(methods)*len(cost_subsets), len(theta_0),samples_per_profile) )
    ranges = np.zeros( (len(noise_mods)*len(dts)*len(methods)*len(cost_subsets), len(theta_0),samples_per_profile) )

    theta_res_history = []
    best_fun_history = []
    idx = 0

    
    for idx_method in range(len(methods)):
        method = methods[idx_method]
        for idx_cost in range(len(cost_subsets)):
            cost_subset = cost_subsets[idx_cost]
            for dt_idx in range(len(dts)):
                dt = dts[dt_idx]
                for idx_noise in range(len(noise_mods)):
                    noise_mod = noise_mods[idx_noise]
                    #    Noise in the measurements
                    sigma_prey=100.0*noise_mod
                    sigma_hunter=3.0*noise_mod
                    bias_prey = 1000*noise_mod
                    bias_hunter = 3*noise_mod
                    
                    #    Calibrate model
                    def F(theta,alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement, cost_term_idx):
                        beta,delta = theta
                        
                        t,prey,hunter = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
                        
                        if cost_term_idx is None:
                            return np.square(np.subtract(prey,prey_measurement)).sum() + np.square(np.subtract(hunter,hunter_measurement)).sum()  
                        elif cost_term_idx == 0:
                            # return np.square(np.subtract(prey,prey_measurement)).sum()
                            return np.square(np.subtract(prey,prey_measurement)).sum() + len(prey_measurement)*np.square(np.subtract(np.mean(hunter),np.mean(hunter_measurement)))
                        elif cost_term_idx == 1:
                            # return np.square(np.subtract(hunter,hunter_measurement)).sum()
                            return np.square(np.subtract(hunter,hunter_measurement)).sum() + len(hunter_measurement)*np.square(np.subtract(np.mean(prey),np.mean(prey_measurement)))
                    
                    #    Measurements generation
                    model = LotkaVolterra()
                    t,prey, hunter, prey_measurement,hunter_measurement = model.generate_measurements(sigma_prey=sigma_prey,sigma_hunter=sigma_hunter,
                                                                                                    bias_prey=bias_prey,bias_hunter=bias_hunter,
                                                                                                      p_0=p_0,h_0=h_0,alpha=alpha_gt,beta=beta_gt,gamma=gamma_gt,delta=delta_gt,T=T,dt=dt)
                    # model.plot_prey_hunter_measurements(t, prey, hunter, prey_measurement, hunter_measurement)
                    theta_names = ['beta','delta']
                        
                    res = minimize(F, theta_0, args=(alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,hunter_measurement, cost_subset), 
                                method=method,   
                                options={'disp': True, 'gtol':1E-8, 'maxls':50})

                    theta_res = res.x
                    best_fun = res.fun
                    print(theta_res)
                    theta_res_history.append(theta_res)
                    best_fun_history.append(best_fun)

                    # plot best fit result
                    t,prey,hunter = model.predict(p_0,h_0,alpha_gt,theta_res[0],gamma_gt,theta_res[1],T,dt)
                    plt.figure(figsize=(20, 5))
                    plt.subplot(1, 3, 1)
                    if cost_subset is None:
                        plt.plot(t, prey, label='Prey')
                        plt.plot(t, prey_measurement, 'o', label='Prey Measurement')
                    elif cost_subset == 0:
                        plt.plot(t, prey, label='Prey')
                        plt.plot(t, prey_measurement, 'o', label='Prey Measurement')
                    elif cost_subset == 1:
                        # plot prey as mean as horizontal line
                        plt.axhline(y=np.mean(prey), color='r', linestyle='--', label='Prey Mean')
                        plt.axhline(y=np.mean(prey_measurement), color='g', linestyle='--', label='Prey Measurement Mean')
                        plt.plot(t, prey, label='Prey')
                        plt.plot(t, prey_measurement, 'o', color='grey', label='unused Prey Measurement')
                        
                    plt.xlabel('Time', fontsize=14)
                    plt.ylabel('Prey Population', fontsize=14)
                    plt.legend()
                    plt.subplot(1, 3, 2)
                    if cost_subset is None:
                        plt.plot(t, hunter, label='Hunter')
                        plt.plot(t, hunter_measurement, 'o', label='Hunter Measurement')
                    elif cost_subset == 0:
                        # plot hunter as mean
                        plt.axhline(y=np.mean(hunter), color='r', linestyle='--', label='Hunter Mean')
                        plt.axhline(y=np.mean(hunter_measurement), color='g', linestyle='--', label='Hunter Measurement Mean')
                        plt.plot(t, hunter, label='Hunter')
                        plt.plot(t, hunter_measurement, 'o', color='grey', label='unused Hunter Measurement')
                    elif cost_subset == 1:
                        plt.plot(t, hunter, label='Hunter')
                        plt.plot(t, hunter_measurement, 'o', label='Hunter Measurement')
                        
                    plt.xlabel('Time', fontsize=14)
                    plt.ylabel('Hunter Population', fontsize=14)
                    plt.legend()
                    plt.subplot(1, 3, 3)
                    plt.plot(prey, hunter, label='Hunter vs Prey')
                    if cost_subset is None:
                        plt.plot(prey_measurement, hunter_measurement, 'o', label='Hunter vs Prey Measurement')
                    plt.xlabel('prey population', fontsize=14)
                    plt.ylabel('hunter population', fontsize=14)
                    plt.legend()
                    plt.savefig(f'lotka_volterra_best_fit_dt_{dt}_noise_mod_{noise_mod}_method_{method}_cost_subset_{cost_subset}.png')
                    plt.close()
                    # plt.show()
                    
                    #    Modified objective function for profile likelihood
                    def F_likelihood(theta, alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement, param_idx, current_value, cost_term_idx):
                        theta_full = np.insert(theta,param_idx,current_value)
                        return F(theta_full,alpha,gamma,p_0,h_0,model,prey_measurement,hunter_measurement, cost_term_idx)
                    
                    #    Iterating over each variable and each sample of the profile
                    for param_idx in range(0,len(theta_res)):
                        theta_init = np.delete(theta_res,param_idx)
                        
                        param_value = theta_res[param_idx]
                        min_range = min_factor * param_value 
                        max_range = max_factor * param_value
                        
                        param_range = np.linspace(min_range,max_range, samples_per_profile)
                        ranges[idx, param_idx,:] = param_range
                        for idx_sample, current_value in enumerate(param_range): 
                            res = minimize(F_likelihood, theta_init, 
                                        args=(alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,
                                                hunter_measurement,param_idx,current_value, cost_subset), 
                                        method=method, options={'gtol':1E-6})
                            profiles[idx, param_idx,idx_sample] = res.fun
                            # print(res)
                    
                    idx += 1
                    # print(profiles)
    
    #    Print profiles
    dt_colors = ['blue', 'orange', 'green']
    noise_linestyles = ['-', '--', '-.']
    for ylim in [6e4, 6e8]:
        res_idx = 0
        for idx_method in range(0, len(methods)):
            for idx_cost in range(0, len(cost_subsets)):
                plt.figure(figsize=(20, 10))
                for idx_dt in range(0,len(dts)):
                    for idx_noise in range(0,len(noise_mods)):
                        for idx_param in range(0,len(theta_0)):
                            
                            plt.subplot(1, len(theta_0), idx_param+1)
                            plt.plot(ranges[res_idx, idx_param,:], profiles[res_idx, idx_param,:], 
                                    color=dt_colors[idx_dt], linestyle=noise_linestyles[idx_noise],
                                    label=f'noise mod = {noise_mods[idx_noise]}, dt = {dts[idx_dt]}')
                            plt.xlabel(theta_names[idx_param], fontsize=14)
                            plt.ylabel('cost', fontsize=14)
                            plt.legend()
                            plt.ylim(0.0,ylim)
                            plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                            plt.tick_params(axis='both', which='major', labelsize=14)


                            # plot over with best fit
                            plt.plot(theta_res_history[res_idx][idx_param], best_fun_history[res_idx], 
                                    color=dt_colors[idx_dt], linestyle='None', label='Best Fit', markersize=14, 
                                    marker='x')
                        res_idx += 1
                # plot a vertical line at the ground truth
                plt.subplot(1, len(theta_0), 1)
                plt.axvline(x=theta_gt[0], color='black', linestyle='--', label='Ground Truth')
                plt.subplot(1, len(theta_0), 2)
                plt.axvline(x=theta_gt[1], color='black', linestyle='--', label='Ground Truth')
                plt.savefig(f'lotka_volterra_profile_likelihood_' + \
                            f'method_{methods[idx_method]}_cost_subset_{cost_subsets[idx_cost]}_ylim_{ylim}.png')
                plt.close()
        
        