'''
Created on 18/05/2025

@summary: Academic example of a Pareto optimisation for unknown parameters in the Lotka-Volterra equation.

@author: Gonzalo D. Maso Talou
'''
from models.LotkaVolterraModel import LotkaVolterra
import numpy as np
import matplotlib.pyplot as plt

#    Initial conditions
p_0=5000
h_0=10
T=14
dt=0.25

#    Ground truth model parameters
alpha = 1.0
beta = 5E-2
gamma = 0.25
delta = 1E-4

#    Noise in the measurements
sigma_prey=100.0
sigma_hunter=3.0

#    Measurements generation
model = LotkaVolterra()
t,prey, hunter, prey_measurement,hunter_measurement = model.generate_measurements(sigma_prey=sigma_prey,sigma_hunter=sigma_hunter,p_0=p_0,h_0=h_0,alpha=alpha,beta=beta,gamma=gamma,delta=delta,T=T,dt=dt)
model.plot_prey_hunter_measurements(t, prey, hunter, prey_measurement, hunter_measurement)

#    Objective definitions
def F1(z,z_hat):
    return np.square(np.subtract(z,z_hat)).mean()

def F2(z,z_hat):
    return np.square(np.subtract(z,z_hat)).mean()

#    Plot F1 and F2 for (beta, delta) pairs
betas = np.linspace(0.025,0.075,50)
deltas = np.linspace(5E-5,15E-5,50)

X, Y = np.meshgrid(betas, deltas)
F1_vals = np.zeros(X.shape)
F2_vals = np.zeros(X.shape)

for i in range(0,X.shape[0]):
    for j in range(0,X.shape[1]):
        t,prey,hunter = model.predict(p_0, h_0, alpha, X[i,j], gamma, Y[i,j], T, dt)
        F1_vals[i,j] = F1(prey,prey_measurement)
        F2_vals[i,j] = F2(hunter,hunter_measurement)

#    Parameter space
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, np.log(F1_vals),cmap=plt.cm.viridis)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\delta$')
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\mathcal{F}_1$')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, np.log(F2_vals),cmap=plt.cm.viridis)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\delta$')
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\mathcal{F}_2$')

plt.show()

#    Functional space
plt.scatter(F1_vals.flatten(),F2_vals.flatten(),color='red', marker='o', label=r'$\hat{z}_{hunter}$', zorder=3)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$\mathcal{F}_1$')
plt.ylabel(r'$\mathcal{F}_2$')
plt.show()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

#    Pymoo - Pareto front calculation

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         xl=np.array([1E-2,1E-6]),
                         xu=np.array([1E0,20E-5]))

    def _evaluate(self, x, out, *args, **kwargs):
        t,prey,hunter = model.predict(p_0, h_0, alpha, x[0], gamma, x[1], T, dt)
        
        f1 = F1(prey,prey_measurement)
        f2 = F2(hunter,hunter_measurement)

        out["F"] = [f1, f2]


problem = MyProblem()

algorithm = NSGA2(pop_size=300)

res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               verbose=False,
               seed=1)

plot = Scatter()
plot.add(res.F, edgecolor="red", facecolor="none")
plot.show()

plt.scatter(res.X[:,0],res.X[:,1],color='red', marker='o', label=r'$\hat{z}_{hunter}$', zorder=3)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\delta$')
plt.show()
