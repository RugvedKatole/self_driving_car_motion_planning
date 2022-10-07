# !usr/bin/env python3
# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 19 September 2022


import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from constraint_limits import PCC_parameters

 #x_f
params = PCC_parameters()

N = int(params.x_f/params.x_d)
ymin = np.array([0]*N)
ymax = np.array([0]*N)

def vehicle_model(z,u,x_d=1):
    
    v = z[0] + u[0]*x_d
    y = z[1] + u[1]*x_d

    return np.array([v,y])

def get_ref(x_tilde):
    
    vr  = np.array([0.0]*N)
    y   = np.array([0.0]*N)

    for i in range(N):
        vr[i] = params.vr - params.vl 
        if params.XL0 - params.lLF <= x_tilde + params.x_d*i and x_tilde + params.x_d*i <= params.XL0 + params.lLr:
            y[i] = 3*params.wl/2
        else:
            y[i] = params.wl/2

    z_ref = np.array([vr,y])
    return z_ref

def MPC(z_initial,U_prev,z_r,Q,R,S):
    """MPC solver"""
    z_e = cvxpy.Variable((2,N),"z")
    u_e = cvxpy.Variable((2,N),"u")
    t_dash = cvxpy.Variable((1,N),"t_dash")
    cost = 0
    
    constraints = [z_e[:,0] == z_initial.flatten()] # 19e

    vr_tilde = params.vr - params.vl

    for i in range(N-1):

        if i != 0:
            cost += cvxpy.quad_form(u_e[:,i+1]-u_e[:,i],S)
        else:
            cost += cvxpy.quad_form(u_e[:,i]-U_prev[:,i],S)
        constraints += [z_e[:,i+1] == np.eye(2) @ z_e[:,i] + np.eye(2)*params.x_d @ u_e[:,i]]  #19 b
    # cost += params.epsilon*t_dash[0,-1]*N
    for i in range(N):
        cost += cvxpy.quad_form(z_e[:,i]-z_r[:,i], Q)
        cost += cvxpy.quad_form(u_e[:,i], R)

        # constraints += [t_dash[0,i] >= cvxpy.inv_pos(z_e[0,i])]

        if  params.x_d*i > params.XL0 - params.lLF and params.x_d*i < params.XL0 + params.lLr:
            ymin[i] = params.wl + params.w
        else:
            ymin[i] = params.w

        if  params.x_d*i > params.XL0 - params.ls and params.x_d*i < params.XL0 + params.le:
            ymax[i] = 2*params.wl - params.w

            # constraints += [(params.x_d*i -params.XO0 -(params.vo-params.vl)*t_dash[0,i]*i)/params.lOf + (z_e[1,i] - params.YO0)/params.wl <= -1]
            # constraints += [(params.x_d*i -params.XA0 -(params.va-params.vl)*t_dash[0,i]*i*params.x_d)/params.lAr - (z_e[1,i] - params.YA0)/params.wl >= 1]

        else:
            ymax[i] = params.wl - params.w 

        constraints += [z_e[0,i] >= params.epsilon]             #19c
        constraints += [z_e[0,i] <= params.vxmax-params.vl]    #19c
        constraints += [z_e[1,i] >= ymin[i]]                       #19c
        constraints += [z_e[1,i] <= ymax[i]]                       #19c

        constraints += [u_e[0,i] >= params.axmin*(2 - z_e[0,i]/vr_tilde)/vr_tilde]  #19d
        constraints += [u_e[0,i] <= params.axmax*(2 - z_e[0,i]/vr_tilde)/vr_tilde]  #19d
        constraints += [u_e[1,i] >= params.vymin*(2 - z_e[0,i]/vr_tilde)/vr_tilde]  #19d
        constraints += [u_e[1,i] <= params.vymax*(2 - z_e[0,i]/vr_tilde)/vr_tilde]  #19d

        constraints += [u_e[1,i] >= params.smin*(1 + params.vl/vr_tilde*(2 - z_e[0,i]/vr_tilde ))]  #19f
        constraints += [u_e[1,i] <= params.smax*(1 + params.vl/vr_tilde*(2 - z_e[0,i]/vr_tilde ))]  #19f


    qp = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    qp.solve(solver=cvxpy.ECOS, verbose=False)
    print(qp.status)
    if qp.status == cvxpy.OPTIMAL or qp.status == cvxpy.OPTIMAL_INACCURATE:
        v = np.array(z_e.value[0,:]).flatten()
        y = np.array(z_e.value[1,:]).flatten()
        
        v_dash = np.array(u_e.value[0,:]).flatten()
        y_dash = np.array(u_e.value[1,:]).flatten()
    # else:
    #     y, v, v_dash, y_dash = None, None, None, None
    #     a, delta = None, None

    return [v,y] , [v_dash,y_dash]

z_des = get_ref(0)
z,u = MPC(z_initial=np.array([[20/3.6],[7.5]]),U_prev=np.array([[0],[0]]),z_r=z_des,Q=params.Q,R=params.R ,S=params.S)


x = np.array([0]*N)
for i in range(N):
    x[i] = i*params.x_d + params.vl*i*params.x_d/z[0][i]

fig, ax = plt.subplots(3,1)
y = np.array([5]*N)


ax[0].plot(list(range(N)),y,"--",color="0.9")
ax[0].plot(list(range(N)),z_des[1,:],"-.",color = "k",linewidth=0.8)
ax[0].plot(list(range(N)),ymin,"-",color = "b",linewidth=0.8)
ax[0].plot(list(range(N)),ymax,"-",color = "b",linewidth=0.8)
ax[0].plot(list(range(N)),z[1],color="k")
ax[0].set_xlim(0,N)
ax[0].set_ylim(0,10)
ax[0].set_ylabel("lateral position(m)")
ax[0].set_xlabel("Relative Longitudinal position(m)")
ax[0].plot(params.XL0,2.5,"s",color="r")
ax[0].plot(80,2.5,"s",color="r")
# ax[0].plot([0,35,60,87,112],[z[1][0],z[1][35],z[1][60],z[1][87],z[1][112]],'o',color="b")

ax[1].plot(x,z[1],color="k")
# ax[1].set_xlim(0,600)
ax[1].set_ylim(0,10)
ax[1].set_ylabel("lateral position(m)")
ax[1].set_xlabel("Absolute Longitudinal position(m)")
ax[1].plot(x,y,"--",color="0.1")


ax[2].plot(x,z[0]*3.6+params.vl*3.6,color="r")
# ax[2].set_xlim(0,600)
ax[2].set_ylim(68,76)
ax[2].set_ylabel("Longitudinal Speed (km/h)")
ax[2].set_xlabel("Absolute Longitudinal position(m)")
# ax[2].plot(x,y,"--",color="0.5")
# ax[2].plot(x,z_des[1,:],"-.",)
# ax[2].plot(x,y,linestyle= "-.",color="k")

ax1 = ax[2].twinx()
ax1.plot(x, u[1]*params.vr, linestyle= "--", color="k")
ax1.set_ylabel("lateral Speed")
ax1.set_ylim(-5,5)


fig.tight_layout()
plt.show()