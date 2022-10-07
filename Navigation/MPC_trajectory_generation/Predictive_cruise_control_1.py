# !usr/bin/env python3
# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 12 September 2022

from math import cos, pi, sin, atan
from statistics import mode
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import ecos

#parameters
X_D = 1
x_f = 200  # prediction horizon
N = 50  #prediction horizon steps

x = MX.sym("x")  # x-co ordinate
y = MX.sym("y")  # y co-ordinate
vx = MX.sym("vx")  # velocity in x



states = vertcat(x,vx,y)
n_states = 2

ax = MX.sym("ax")  # acceleration in x
vy = MX.sym("vy")  # velocity in y
vx_dash = MX.sym("vx_dash")
y_dash = MX.sym("vy_dash")

n_control = 2
controls = vertcat(ax,vy)

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
beta = 50*pi/180
smax = atan(beta)
smin = -atan(beta)


epsilon = 0.01

#weights 
Q = np.diag([10,0.1,0.001])
R = np.diag([1,1])
S = np.diag([1,1])


def get_ref(type="A"):
    if type == "A":
        vr  = np.array([0.0]*x_f*2)
        y   = np.array([0.0]*x_f*2)
        x   = np.array([0.0]*x_f*2)
        for i in range(x_f*2):
            vr[i] = 20/3.6
            x[i]  = i
            if XLO - l_Lf <= i and i <= XLO + l_Lr:
                y[i] = 3*w_l/2
            else:
                y[i] = w_l/2

        z_ref = np.array([x,vr,y])
    return z_ref

# point mass model
model = vertcat((vx),(ax),(vy))

model_estimator = Function("Plant_model", [states, controls], [model])

#integrating our ODE for t = 0.2
intg_options = {"tf":0.18,"simplify":True, "number_of_finite_elements":4}

DAE = {"x":states, "p": controls, "ode":model_estimator(states,controls)}

intg = integrator("intg","rk",DAE,intg_options)

res = intg(states,controls,[],[],[],[])
States_nxt = res[0]
model_estimator = Function("Estimator",[states,controls],[States_nxt])

# sim = model_Estimator.mapaccum(180)

# u = np.array([[0]*180,[0]*180])
# res = sim([0,20/3.6,2.5],u)

# print(res)

opti = Opti()

z         = opti.variable(3,N+1)
u         = opti.variable(2,N)
z_initial = opti.parameter(3,1)
u_prev    = opti.parameter(2,1)

z_ref     = opti.parameter(3,N+2)

obj=0
for i in range(N-1):
    obj += (z[:,i+1]-z_ref[:,i]).T @ Q @ (z[:,i+1]-z_ref[:,i])  # states objective
    obj += (u[:,i+1]-u[:,i]).T/0.18 @ S @ (u[:,i+1]-u[:,i])/0.18 # change in control variables
    obj += u[:,i].T @ R @ u[:,i] # control variables

opti.minimize(obj)

opti.subject_to(z[:,0]==z_initial)
for k in range(N):
    opti.subject_to(z[:,k+1]==model_estimator(z[:,k],u[:,k]))

ymin = np.array([0.0]*N)
ymax = np.array([0.0]*N)
for i in range(N):
    opti.subject_to(z[0,i]>0)
    opti.subject_to(z[1,i ]>= epsilon)
    opti.subject_to(z[1,i] <= vxmax - vl)

    if XLO - l_Lf <= i and i <= XLO + l_Lr:
        ymin[i] = w_l + w
    else:
        ymin[i] = w

    if XLO - ls <= i and i <= XLO + le:
        ymax[i] = 2*w_l - w
    else:
        ymax[i] = w_l - w

    opti.subject_to(z[2,i] >= ymin[i])
    opti.subject_to(z[2,i] <= ymax[i])

    # rate of control change constraints
    opti.subject_to(u[0,i] >= axmin)
    opti.subject_to(u[0,i] <= axmax)
    opti.subject_to(u[1,i] >= vymin)
    opti.subject_to(u[1,i] <= vymax)

    # opti.subject_to(u[1,i] >= smin*(z[1,i]))
    # opti.subject_to(u[1,i] <= smax*(z[1,i]))

opts = { 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-6, 'ipopt.acceptable_obj_change_tol':1e-6}
opti.solver('ipopt')

MPC = opti.to_function("MPC",[z_initial,u_prev,z_ref],[u[:,1]])

z = np.array([0,20/3.6,w_l/2])
u_prev = np.array([0.0,0.0])
z_ref = get_ref("A")
print(z_ref.shape)
control_log = np.array([u_prev])
States_log = np.array([z.T])

for i in range(190):
    # print(z_ref[:,0:12].shape)
    u = MPC(z,u_prev,z_ref[:,0:N+2])
    print(i)
    u_prev = np.array(u)
    # simulate system
    print(z,u)
    z = model_estimator(z,u)
    z_ref = np.delete(z_ref,0,axis=-1)
    if z[0,0] >180:
        break
    z=np.array(z)
    control_log = np.append(control_log,u_prev.T,axis=0)
    States_log = np.append(States_log,z.T,axis=0)

print(control_log.shape)
print(States_log.shape)

States_log = States_log.T
control_log = control_log.T
z_ref = get_ref()
plt.plot(States_log[0,:],States_log[2,:])
plt.plot(States_log[0,:],np.array([5]*States_log.shape[1]),"--",color="k")
plt.plot(ymin)
plt.plot(ymax)
plt.xlim([0,180])
plt.ylim([0,10])

# plt.plot(z_ref[2,0:180],"--","b")
plt.show()

