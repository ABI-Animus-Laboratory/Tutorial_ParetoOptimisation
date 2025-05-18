'''
Created on 18/05/2025

@summary: Example presenting the Lotka-Volterra (prey-hunter populational) model.

@author: Gonzalo D. Maso Talou
'''
from models.LotkaVolterraModel import LotkaVolterra

#    Initial conditions
p_0=5000
h_0=10
T=28
dt=0.5

#    Model parameters
alpha = 1.0
beta = 0.05
gamma = 0.1
delta = 0.0001

#    Model creation and prediction
model = LotkaVolterra()
t,prey,hunter = model.predict(p_0,h_0,alpha,beta,gamma,delta,T,dt)

model.plot_prey_hunter(t, prey, hunter)

#    Measurements generation
t,prey,hunter,prey_measurement,hunter_measurement = model.generate_measurements(sigma_prey=15,sigma_hunter=1.5,p_0=p_0,h_0=h_0,alpha=alpha,beta=beta,gamma=gamma,delta=delta,T=T,dt=dt)

model.plot_prey_hunter_measurements(t,prey,hunter,prey_measurement,hunter_measurement)

