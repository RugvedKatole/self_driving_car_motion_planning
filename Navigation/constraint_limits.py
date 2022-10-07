from math import atan, pi
import numpy as np


class limit(object):
    """limits for different constraint parameters """
    
    vx_max = 20
    vx_min = 0

    vy_max = 5
    vy_min = -5

    ax_max = 1
    ax_min = -4

    ay_max = 2
    ay_min = -2

    jx_max =  1.5
    jx_min = -3

    jy_max = 0.5
    jy_min = -0.5

    y_max = 7.5
    y_min = -2.5


class PCC_parameters(object):
    "parameters and limits for PCC paper"
    x_d = 1
    x_f = 180
    wl  = 3.5
    w   = 0.75
    ls  = 40
    le  = 37.3
    lLF = 15
    lLr = 12.3
    lOf = 48.4
    lAr = 9.5 
    
    XE0 = 0
    XL0 = 75
    XO0 = 650
    XA0 = 0
    YE0 = 2.5
    YL0 = 2.5
    YO0 = 7.5
    YA0 = 7.5

    vr = 70/3.6
    vl = 50/3.6
    vo = -70/3.6
    va = 70/3.6


    vxmax = 10/3.6
    vymin = -4
    vymax = 4
    axmin = -4
    axmax = 1
    smin = -atan(10*pi/180)
    smax = atan(10*pi/180)
    epsilon = 0.01

    Q = np.diag([0.01,0.1])
    R = np.diag([2,20])
    S = np.diag([100,400])
