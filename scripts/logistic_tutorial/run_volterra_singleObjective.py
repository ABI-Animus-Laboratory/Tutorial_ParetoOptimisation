'''
Created on 21/05/2025

@summary: Example of single-objective optimisation.

@author: Gonzalo D. Maso Talou
'''

from models.LotkaVolterraModel import LotkaVolterra

import numpy as np
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
    def F(theta,args):
        beta,delta = theta
        alpha, gamma, p_0, h_0,model,prey_measurement,hunter_measurement = args
        t,prey,hunter = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)
        
        return np.square(np.subtract(prey,prey_measurement)).mean() + np.square(np.subtract(hunter,hunter_measurement)).mean()  
         
    res = minimize(F, [1E-2,1E-2], args=[alpha_gt,gamma_gt,p_0,h_0,model,prey_measurement,hunter_measurement], method='BFGS',
                   options={'disp': True})
    
    print(res)