# !usr/bin/env python3
# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 12 September 2022

from math import cos, pi, sin, atan
import numpy as np
from casadi import *
import matplotlib.pyplot as plt

#parameters
X_D = 1
x_f = 180  # prediction horizon
N = int(x_f/X_D)  #prediction horizon steps

x = MX.sym("x")  # x-co ordinate
y = MX.sym("y")  # y co-ordinate
vx = MX.sym("vx")  # velocity in x



states = vertcat(vx,y)
n_states = 2

ax = MX.sym("ax")  # acceleration in x
vy = MX.sym("vy")  # velocity in y
vx_dash = MX.sym("vx_dash")
y_dash = MX.sym("vy_dash")

n_control = 2
controls = vertcat(vx_dash,y_dash)

# bounds and other parameters
# x_f = 180 # m
w_l = 5  # m
w   = 1.5
ls  = 40
le  = 37.3
l_Lf= 15
l_Lr= 12.3
l_Of= 48.4
l_Ar= 9.5

XE0 = 0
XLO = 75
XO0 = 650
XA0 = 0
YE0 = 2.5
YL0 = 2.5
YO0 = 7.5
YA0 = 7.5

vr = 70 #km/hr
vl = 50
vo = -70
va = 70
vxmax = 80
vymin = -4
vymax = 4
axmin = -4
axmax = 1
beta = 10*pi/180
smax = atan(beta)
smin = -atan(beta)


epsilon = 0.01

#weights 
Q = np.diag([0.01,0.1])
R = np.diag([2,20])
S = np.diag([100,400])


def get_ref(type):
    if type == "A":
        vr  = np.array([0.0]*N*2)
        y   = np.array([0.0]*N*2)
        for i in range(N*2):
            vr[i] = 70
            if XLO - l_Lf <= i and i <= XLO + l_Lr:
                y[i] = 3*w_l/2
            else:
                y[i] = w_l/2

        z_ref = np.array([vr,y])
    return z_ref

 ## first order point mass model
model = vertcat((vx),(ax),(vy))

# model linearised

model_lin = vertcat((vx_dash),(y_dash)) # states_dash = controls  ## dash is derivative w.r.t x(distance)

Estimator = Function("Estimator", [states,controls],[model_lin])

## creating an estimator from model
intg_options = {"tf": X_D, "simplify":True, "number_of_finite_elements":4}


DAE = {"x": states, "p": controls, "ode":Estimator(states,controls)}

intg = integrator("intg","rk", DAE, intg_options)

res = intg(states, controls, [],[],[],[])
states_nxt = res[0]

Estimator = Function("Estimator",[states,controls],[states_nxt])


opti = Opti()

z         = opti.variable(2,N+1)
u         = opti.variable(2,N)
z_initial = opti.parameter(2,1)
u_prev    = opti.parameter(2,1)

z_ref     = opti.parameter(2,N+2)

obj = 0

for i in range(N-1):
    obj += (z[:,i+1]-z_ref[:,i]).T @ Q @ (z[:,i+1]-z_ref[:,i])
    obj += ((u[:,i+1]-u[:,i])/X_D).T @ S @ ((u[:,i+1]-u[:,i])/X_D) # change in control variables w.r.t distance
    obj += u[:,i].T @ R @ u[:,i] # control variables


opti.minimize(obj)

opti.subject_to(z[:,0]==z_initial)
for k in range(int(x_f/X_D)):
    opti.subject_to(z[:,k+1]==Estimator(z[:,k],u[:,k]))

for i in range(int(x_f/X_D)):

    # state constraints
    opti.subject_to(z[0,i] >= epsilon)
    opti.subject_to(z[0,i] <= vxmax - vl)
    #getting y min limits
    
    if XLO + -l_Lf <= i and i <= XLO + l_Lr:
        ymin = w_l + w
    else:
        ymin = w

    if XLO + -ls <= i and i <= XLO + le:
        ymax = 2*w_l - w
    else:
        ymax = w_l - w

    opti.subject_to(z[1,i] >= ymin)
    opti.subject_to(z[1,i] <= ymax)

    # rate of control change constraints
    opti.subject_to(u[0,i] >= axmin/z[0,i])
    opti.subject_to(u[0,i] <= axmax/z[0,i])
    opti.subject_to(u[1,i] >= vymin/z[0,i])
    opti.subject_to(u[1,i] <= vymax/z[0,i])

    opti.subject_to(u[1,i] >= smin*(1+vl/z[0,i]))
    opti.subject_to(u[1,i] <= smax*(1+vl/z[0,i]))


# solver_options = {"print_header":False,"print_iteration": False, "print_time": False, "print_in":False,
#                     "print_out":False}
opts = { 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-6, 'ipopt.acceptable_obj_change_tol':1e-6}
opti.solver('ipopt')

# opti.set_value(z_initial,[5,0,1,0])
# sol = opti.solve()

MPC = opti.to_function("MPC",[z_initial,u_prev,z_ref],[u[:,1]])
# print(MPC)
# MPC loop

z = np.array([5.56,w_l/2])
u_prev = np.array([0.02,0.02])
z_ref = get_ref("A")
print(z_ref.shape)
control_log = np.array([u_prev])
States_log = np.array([z.T])

for i in range(N-1):
    # print(z_ref[:,0:12].shape)
    u = MPC(z,u_prev,z_ref[:,0:N+2])
    print(i)
    u_prev = np.array(u)
    # simulate system
    z = Estimator(z,u)
    z_ref = np.delete(z_ref,0,axis=-1)

    z=np.array(z)
    control_log = np.append(control_log,u_prev.T,axis=0)
    States_log = np.append(States_log,z.T,axis=0)

print(control_log.shape)
print(States_log.shape)

States_log = States_log.T
control_log = control_log.T

plt.plot(list(range(0,N-1,1)),States_log[1][list(range(0,N-1,1))])
plt.show()
