# !usr/bin/env python3
# Copyright (c) @ RugvedKatole
#
# Author: Rugved Katole
# Affliation: Indian Institute of Bombay
# Date: 14 September 2022

"""@INPROCEEDINGS{6728267,
  author={Nilsson, Julia and Ali, Mohammad and Falcone, Paolo and Sjöberg, Jonas},
  booktitle={16th International IEEE Conference on Intelligent Transportation Systems (ITSC 2013)}, 
  title={Predictive manoeuvre generation for automated driving}, 
  year={2013},
  volume={},
  number={},
  pages={418-423},
  doi={10.1109/ITSC.2013.6728267}}"""



from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from constraint_limits import limit

N = 10  # prediction horizon
t = 0.1 # sampling time

z = MX.sym("state_vector",4,1)
u = MX.sym("control_v",2,1)

point_mass_model = vertcat(z[2],z[3],u[0],u[1])
vehicle_model = Function("pm",[z,u],[point_mass_model])
intg_options = {"tf": t}
DAE = {"x":z, "p":u, "ode":vehicle_model(z,u)}
intg = integrator("intg","cvodes",DAE,intg_options)
res = intg(z, u, [],[],[],[])
states_nxt = res[0]

vehicle_model = Function("pm",[z,u],[states_nxt])

# defining our states and models
limits = limit()
# weights
alpha = 1
kappa = 0.5
gamma = 1
nu    = 0.5
rho   = 0.5
xi    = 10000
chi   = 10000

theta_f = 2
theta_r = 1
Lc = 5
WL = 5
WC = 2.5 
sigma = 5
psi = 2.5



opti = Opti('nlp')


z_e = opti.variable(4,N)  # ego vehicle state vector [delta_x,y,vx,vy]
u_e = opti.variable(2,N)    # ego vehicle control vector [ax,ay]

E_f = opti.variable(1,N)   #slack variables epsilon front and rear
E_r = opti.variable(1,N)   

z_s = opti.parameter(4,N) # surrounding vehicle states (x,y,vx,vy)
z_des = opti.parameter(2,N) # desired velocity and y_ref 

FCC = True
u_prev    = opti.parameter(2,1) # previous control vector
z_initial = opti.parameter(4,1) # initial state

#relative parameters
delta_x = z_s[0,:] - z_e[0,:]
delta_y = z_s[1,:] - z_e[1,:]
delta_v = z_s[2,:] - z_e[2,:]


# now defining objective function
obj0 = 0

for i in range(N):

    obj0 += alpha*(z_e[2,i] - z_des[0,i])**2 + kappa*(z_e[2,i] - z_des[0,i])**2
    obj0 += gamma*z_e[3,i]**2 + nu*u_e[0,i]**2 + rho*u_e[1,i]**2



# approach 1
obj1 = 0
for i in range(N):
  obj1 += z_e[0,i]*E_f[0,i] - z_e[0,i]*E_r[0,i]

obj1 += obj0


# # approach 2 comment approach 1 before uncommenting this
# obj2 = 0
# for i in range(N):
#   obj2 += chi*E_f[0,i]**2 + xi*E_r[0,i]**2

# obj2 += obj0

opti.minimize(obj0)

#initial constraint X(0) = x(t)
opti._subject_to(z_e[0,i] == z_initial[0,0] )
opti._subject_to(z_e[1,i] == z_initial[1,0] )
opti._subject_to(z_e[2,i] == z_initial[2,0] )
opti._subject_to(z_e[3,i] == z_initial[3,0] )

#Constraints
for i in range(N-1):
  #constraint no. 7
  opti._subject_to(z_s[0,i+1] - z_e[0,i+1] == z_s[0,i] - z_e[0,i] + (z_e[2,i]-z_s[2,i])*t)
  opti._subject_to(z_e[1,i+1] == z_e[1,i] + (z_e[3,i])*t)
  opti._subject_to(z_e[2,i+1] == z_e[2,i] + (u_e[0,i])*t)
  opti._subject_to(z_e[3,i+1] == z_e[3,i] + (u_e[1,i])*t)

for i in range(N):

  #constraint no. 8
  opti._subject_to(z_e[3,i] <= 0.17*z_e[2,i])
  opti._subject_to(0.17*z_e[2,i] <= z_e[3,i])

  #constraint no. 9

  opti._subject_to(z_e[2,i] <= limits.vx_max)   # 9a
  opti._subject_to(z_e[2,i] >= limits.vx_min)

  opti._subject_to(z_e[1,i] <= limits.y_max)    # 9b
  opti._subject_to(z_e[1,i] >= limits.y_min)

  opti._subject_to(z_e[3,i] <= limits.vy_max)   # 9c
  opti._subject_to(z_e[3,i] >= limits.vy_min)

  opti._subject_to(u_e[0,i] <= limits.ax_max)   # 9d
  opti._subject_to(u_e[0,i] >= limits.ax_min)

  opti._subject_to(u_e[1,i] <= limits.ay_max)   # 9e
  opti._subject_to(u_e[1,i] >= limits.ay_min)
  
  if i != 0:
    opti._subject_to(u_e[0,i] - u_e[0,i-1] <= limits.jx_max)   # 9f
    opti._subject_to(u_e[0,i] - u_e[0,i-1] >= limits.jx_min)

    opti._subject_to(u_e[1,i] - u_e[1,i-1] <= limits.jy_max)   # 9g
    opti._subject_to(u_e[1,i] - u_e[1,i-1] >= limits.jy_min)
  else:
    opti._subject_to(u_e[0,i] - u_prev[0,0] <= limits.jx_max)   # 9f
    opti._subject_to(u_e[0,i] - u_prev[0,0] >= limits.jx_min)

    opti._subject_to(u_e[1,i] - u_prev[1,i-1] <= limits.jy_max)   # 9g
    opti._subject_to(u_e[1,i] - u_prev[1,i-1] >= limits.jy_min)



  # FCC and RCC constraints

  Lf = z_e[2,i]*theta_f + Lc

  Lr = z_e[2,i]*theta_r + Lc

  W = 0.5*WL + WC
  if FCC > 0:
    print(FCC)
    opti._subject_to(z_e[0,i]/Lf - (z_e[1,i] - z_s[1,i])/W + E_f[0,i] >= 1)
  if not FCC < 0:
    print("RCC")
    opti._subject_to(z_e[0,i]/Lr + (z_e[1,i] - z_s[1,i])/W + E_r[0,i] <= -1)
  ita = 2*Lf
  opti._subject_to(z_e[0,i] + ita*E_f[0,i] >= 0)
  opti._subject_to(z_e[0,i] + ita*E_r[0,i] <= 0)


opti.set_initial(E_r,1)
opti.set_initial(E_f,1)

opts = { 'ipopt.max_iter':100,'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-8}

opti.solver('ipopt',opts)

MPC = opti.to_function("MPC", [z_s,z_des,u_prev,z_initial], [u_e[:,1]])


#initial conditions

z_initial = np.array([50,0,20,0])
u_initial = np.array([0,0])

z_s_initial = np.array([50,0,10,0])
u_s_initial = np.array([0,0])
def get_ref():
  v_des = np.array([20.0]*N)
  y_ref = np.array([0.0]*N)

  return np.array([v_des,y_ref])

z_s = vehicle_model.mapaccum(N)

z_s_log = np.array([z_s_initial.T])
z_e_log = np.array([z_initial.T])
u_e_log = np.array([u_initial.T])

for i in range(200):
  z_des = get_ref()
  
  u = MPC(z_s(z_s_initial,u_s_initial),z_des,u_initial,z_initial)
  print(u)
  u_initial = u

  z_initial = vehicle_model(z_initial,u_initial)
  print(z_initial)
  z_s_initial = z_s(z_s_initial,u_s_initial)[:,1]
  if z_s_initial[0] - z_initial[0] <0:
    FCC = False



  z_e_log = np.append(z_e_log,z_initial.T,axis=0)
  z_s_log = np.append(z_s_log,z_s_initial.T,axis=0)
  u_e_log = np.append(u_e_log,u_initial.T,axis=0)
  print(i)

delta_x = z_e_log[:,0]
delta_y = z_s_log[:,1] - z_e_log[:,1]


plt.plot(delta_x,delta_y)
plt.plot(z_e_log[:,0],z_e_log[:,2])
plt.show()


