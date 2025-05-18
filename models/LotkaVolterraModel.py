'''
Created on 18/05/2025

@author: Gonzalo D. Maso Talou
'''

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class LotkaVolterra(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
 
    def Lotka_Volterra_ode(self,y,t,alpha,beta,gamma,delta):
        
        p,h = y
        dydt=[alpha*p - beta*p*h, -gamma*h + delta*h*p]
    
        return dydt
    
    #    This is the model f(x,\theta) for our optimisation
    def predict(self,p_0=5000,h_0=10,alpha = 1.0,beta = 0.1,gamma = 0.1,delta = 0.0001, T=24.5,dt=0.25):
    
        Y_0 = [p_0,h_0]
        t = np.linspace(0, T, int(np.round(T/dt)))
        
        sol = odeint(self.Lotka_Volterra_ode,Y_0,t, args=(alpha,beta,gamma,delta))
        
        return [t,sol[:,0], sol[:,1]]
    
    def generate_measurements(self,sigma_prey=5,sigma_hunter=1,p_0=5000,h_0=10,alpha = 1.0,beta = 0.1,gamma = 0.1,delta = 0.0001, T=24.5,dt=0.25):
    
        Y_0 = [p_0,h_0]
        t = np.linspace(0, T, int(np.round(T/dt)))
        
        sol = odeint(self.Lotka_Volterra_ode,Y_0,t, args=(alpha,beta,gamma,delta))
        
        prey = sol[:,0] 
        prey_noise = np.random.normal(0.0, sigma_prey, size = prey.shape)
        prey_measurement = prey + prey_noise
    
        hunter = sol[:,1] 
        hunter_noise = np.random.normal(0.0, sigma_hunter, size = hunter.shape)
        hunter_measurement = hunter + hunter_noise
        
        return [t,prey, hunter, prey_measurement, hunter_measurement]

    def plot_prey_hunter(self,t,prey,hunter):
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.plot(t, prey, label='Prey')
        plt.xlabel('t')
        plt.ylabel('Prey')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(t, hunter, label='Hunter')
        plt.xlabel('t')
        plt.ylabel('Hunter')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(prey, hunter, label='Hunter(Prey)')
        plt.xlabel('Prey')
        plt.ylabel('Hunter')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_prey_hunter_measurements(self,t,prey,hunter,prey_measurement,hunter_measurement):
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.plot(t, prey, label=r'$z_{prey}$')
        plt.scatter(t, prey_measurement, color='red', marker='o', label=r'$\hat{z}_{prey}$', zorder=3)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$z_{prey}$')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(t, hunter, label=r'$z_{hunter}$')
        plt.scatter(t, hunter_measurement, color='red', marker='o', label=r'$\hat{z}_{hunter}$', zorder=3)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$z_{hunter}$')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(prey, hunter, label=r'$(z_{hunter},z_{prey})$')
        plt.scatter(prey_measurement, hunter_measurement, color='red', marker='o', label=r'$(\hat{z}_{hunter},\hat{z}_{prey})$', zorder=3)
        plt.xlabel(r'$z_{prey}$')
        plt.ylabel(r'$z_{hunter}$')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
