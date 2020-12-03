# Copyright (C) 2020, Bernd Arendt
# Copyright (C) 2015-2020, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import ttk

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *
from user_libs.antennas.discretGeo import disGeometryGprMax

def horn_burr(x, y, z, resolution = 0.0005, rotation = 0, source = 'voltage',measurement ='monostatic'):
    """
    Insert a TEM Hornantenna similar to the antenna ..... (insert paperlink)

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna.
        resolution (float): Spatial resolution for the antenna model =< 1 mm.
        rotation (float): Rotate model in degrees CCW in xy plane.
        measurement ='monostatic', 'rx', 'tx'
    """
    L           = 0.2               # m
    L_plate     = 0.07
    #resolution  = 0.0005            # m
    R           = 0.019978589752617
    Rside       = 0.011
    B           = 0.17                   # m
    tol3d       = 0.0003
    G           = 0.0023 + tol3d    # m
    arcRadius   = 0.025            # m
    W1          = 0.13              # m
    wfeed       = 0.012 #+ tol3d          # m
    C1          = ((W1*1e3/2) - (G*1e3/2))/(np.exp(R*L*1e3)-np.exp(R*0))
    C1_side     = (B*1e3/2-wfeed*1e3/2)/(np.exp(Rside*L*1e3)-np.exp(Rside*0))
    C2          = (G*1e3/2 * np.exp(R*L*1e3)-W1*1e3/2 * np.exp(R*0))/(np.exp(R*L*1e3)-np.exp(R*0))  # 0.259141  ( G / 2 mm * exp(R * L / 1 mm) - W1 / 2 mm * exp(0 oE) ) / ( exp(R * L / 1 mm) - exp(0 oE) )
    C2_side     = (wfeed*1e3/2*np.exp(Rside*L*1e3)-B*1e3/2*np.exp(Rside*0))/(np.exp(Rside*L*1e3)-np.exp(Rside*0))
    slopeAngel  = np.arctan(C1*R*np.exp(R*L*1e3))  # 52.291047
    arcAngle    = np.deg2rad(90)                # deg
    arcStart    = (slopeAngel - arcAngle)       # deg
    arcStop     = arcAngle
    arcDelta    = arcStop - arcStart
    Lfeed       = 0.02                  # m
    deltaB      = 0.02                  # m
    B2          = B + deltaB            # m
    lBalun      = 0.014                 # m
    balunDiaBig = 0.012                 # m
    balunDiaSmall = 0.008               # m
    smaDia      = 0.001                 # m
    smaTeflonDia= 0.005                 # m
    smaSocketDia= 0.008                 # m
    smaScrwDia  = 0.006                 # m
    sourceresistance = 50              # ohm
    
    # if resolution == 0.001:
    #     dx = 0.001
    #     dy = 0.001
    #     dz = 0.001
    # else:
    #     raise CmdInputError('This antenna module can only be used with a spatial discretisation of 1mm')
    
    material(2.1, 0, 1, 0, 'smaTeflon')

    # end of balun/begin horn x, y, z center
    sma_center = x, y, z + lBalun

    ### Balun ###
    # top plate
    box(sma_center[0]+(G/2), sma_center[1]-balunDiaBig/2, z, sma_center[0]+(G/2+resolution), sma_center[1]+balunDiaBig/2, sma_center[2],  'pec')
    # bottom plate
    box(sma_center[0]-(G/2+resolution), sma_center[1]-balunDiaSmall/2, z, sma_center[0]-(G/2), sma_center[1]+balunDiaSmall/2, sma_center[2],  'pec')
    triangle(sma_center[0]-(G/2+resolution), sma_center[1]+balunDiaSmall/2, sma_center[2], sma_center[0]-(G/2+resolution),
                sma_center[1]+balunDiaSmall/2, z, sma_center[0]-(G/2+resolution), sma_center[1]+wfeed/2, z, resolution, 'pec')
    triangle(sma_center[0]-(G/2+resolution), sma_center[1]-balunDiaSmall/2, sma_center[2], sma_center[0]-(G/2+resolution), 
                sma_center[1]-balunDiaSmall/2, z, sma_center[0]-(G/2+resolution), sma_center[1]-wfeed/2, z, resolution, 'pec')
    # top big part
    cylinder(sma_center[0]+G/2, sma_center[1], sma_center[2], sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], balunDiaBig/2, 'pec')
    # substract hole 
    cylinder(sma_center[0]+G/2, sma_center[1], sma_center[2], sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], 0.004/2, 'free_space')
    # bottom small part
    cylinder(sma_center[0]-G/2, sma_center[1], sma_center[2], sma_center[0]-(G/2+resolution), sma_center[1], sma_center[2], balunDiaSmall/2, 'pec') 

    ### SMA Connector ###
    if source == 'voltage':
        cylinder(sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.002+resolution, sma_center[1], sma_center[2], smaSocketDia/2, 'pec' )
        cylinder(sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaScrwDia/2, 'pec')
        cylinder(sma_center[0]-(G/2), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaTeflonDia/2, 'smaTeflon')
        cylinder(sma_center[0]-(G/2+resolution+0.001), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaDia/2, 'pec')
        #edge(sma_center[0]-(G/2+0.002), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010, sma_center[1], sma_center[2], 'myTest')

    ### Horn ###
    print('C1:'+str(C1)+' C1side:'+str(C1_side)+' C2:'+str(C2)+' C2side:'+str(C2_side) + ' SlopeAngel:' + str(np.rad2deg(slopeAngel))+'\n'+'ArcStart:'+ str(np.rad2deg(arcStart)))
    zl = np.arange(0, (L+resolution)*1e3, resolution*1e3)
    alpha = np.linspace(arcStart, arcStop, num=200, endpoint=True)
    # horn
    x_z = C1 * np.exp((zl*R))+C2      # horn part
    y_z = C1_side*np.exp(Rside*zl)+C2_side+(wfeed*1e3/2)-wfeed*1e3/2   # c1_side*efk^(R_side*v*u)+c2_side+(w_feed/2*v)-w_feed/2 # C1_SIDE*exp(R_SIDE*t)+C2_SIDE+(W_FEED/2mm)-W_FEED/2
    x_alpha = W1*1e3/2+arcRadius*1e3*np.sin(alpha)+np.cos(slopeAngel)*arcRadius*1e3
    y_alpha = B*1e3/2+deltaB*1e3/2*(alpha-arcStart)/arcDelta #B/2+(DELTA_B)/2*(t-ARC_START/1grd)/(ARC_DELTA/1grd)
    z_alpha = L*1e3+arcRadius*1e3*np.cos(alpha)-np.sin(slopeAngel)*arcRadius*1e3

    # discret horn funcion
    zn, yn_z, xn_z = disGeometryGprMax(zl, y_z, x_z, resolution*1e3)
    za, ya, xa = disGeometryGprMax(z_alpha, y_alpha, x_alpha, resolution*1e3)
    
    ### koordinates in gprMax-koordinates mm --> m ###
    # horn
    zn_real         = z-zn/1e3
    xn_zreal_pos    = x + xn_z/1e3
    xn_zreal_neg    = x - xn_z/1e3
    yn_zreal_pos    = y + yn_z/1e3
    yn_zreal_neg    = y - yn_z/1e3

    # radius
    za_real         = z - za/1e3
    xa_real_pos     = x + xa/1e3
    xa_real_neg     = x - xa/1e3
    ya_real_pos     = y + ya/1e3
    ya_real_neg     = y - ya/1e3
    # plot function
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # #plt.plot(zn_real, yn_zreal_pos, xn_zreal_pos, zn_real, yn_zreal_neg, xn_zreal_neg)
    # #plt.show()
    # plt.plot(za_real, ya_real_pos, xa_real_pos)
    # plt.show()
    
    # build geometry
    # horn
    for n in range(0, len(zn_real)-1):
        #print(zn_real[n], zn_real[n+1])
        xup = np.amax([xn_zreal_pos[n], xn_zreal_pos[n+1]])+resolution
        xlow = np.amin([xn_zreal_pos[n], xn_zreal_pos[n+1]])
        yup = np.amax([yn_zreal_pos[n], yn_zreal_pos[n+1], yn_zreal_neg[n], yn_zreal_neg[n+1]])
        ylow = np.amin([yn_zreal_pos[n], yn_zreal_pos[n+1], yn_zreal_neg[n], yn_zreal_neg[n+1]])
        zup = np.amax([zn_real[n], zn_real[n+1]])+resolution
        zlow = np.amin([zn_real[n], zn_real[n+1]])

        # if xup-xlow and yup-ylow > 0 or xup-xlow and zup-zlow > 0 or yup-ylow and zup-zlow > 0:
        #     plate(xlow, ylow, zlow, xup, yup, zup,'pec')

        if (xup-xlow) and (yup-ylow) and (zup-zlow) > 0:
            box(xlow, ylow, zlow, xup, yup, zup,'pec')

        xup = np.amax([xn_zreal_neg[n], xn_zreal_neg[n+1]])
        xlow = np.amin([xn_zreal_neg[n], xn_zreal_neg[n+1]])-resolution
        # if xup-xlow and yup-ylow > 0 or xup-xlow and zup-zlow > 0 or yup-ylow and zup-zlow > 0:
        #     plate(xlow, ylow, zlow, xup, yup, zup,'pec')

        if (xup-xlow) and (yup-ylow) and (zup-zlow) > 0:
            box(xlow, ylow, zlow, xup, yup, zup,'pec')

    # radius   
    for n in range(0, len(za_real)-1):
        xup = np.amax([xa_real_pos[n], xa_real_pos[n+1]])+resolution
        xlow = np.amin([xa_real_pos[n], xa_real_pos[n+1]])
        yup = np.amax([ya_real_pos[n], ya_real_pos[n+1], ya_real_neg[n], ya_real_neg[n+1]])
        ylow = np.amin([ya_real_pos[n], ya_real_pos[n+1], ya_real_neg[n], ya_real_neg[n+1]])
        zup = np.amax([za_real[n], za_real[n+1]])#+resolution
        zlow = np.amin([za_real[n], za_real[n+1]])-resolution

        # if xup-xlow and yup-ylow > 0 or xup-xlow and zup-zlow > 0 or yup-ylow and zup-zlow > 0:
        #     plate(xlow, ylow, zlow, xup, yup, zup,'pec')

        if (xup-xlow) and (yup-ylow) and (zup-zlow) > 0:
            box(xlow, ylow, zlow, xup, yup, zup,'pec')

        xup = np.amax([xa_real_neg[n], xa_real_neg[n+1]])
        xlow = np.amin([xa_real_neg[n], xa_real_neg[n+1]])-resolution

        if (xup-xlow) and (yup-ylow) and (zup-zlow) > 0:
            box(xlow, ylow, zlow, xup, yup, zup,'pec')
        

    ### Source on SMA Connector ###
    #TODO: find and fix tx solution
    tx = sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2]
    if source == 'voltage':
        print('#waveform: gaussian 1 1e9 myGaussian')
        voltage_source('x', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution))
        cylinder(tx[1], tx[2], tx[3], tx[1]+resolution, tx[2], tx[3], smaScrwDia/2, 'pec')

    if source == 'transmissionline':
        popupmsg('Transmissionline not supportet just jet!!')
        raise CmdInputError('Transmissionline not supportet just jet!!')


    ### Reciever on SMA Connector ###
    if measurement == 'monostatic':
        #rx = tx
        identifier = 'monostatic'
    elif measurement == 'rx':
        identifier = 'rxhorn'
    elif measurement == 'tx':
        identifier = 'txhorn'
    else:
        raise CmdInputError('This antenna have 3 measuremnt methods - tx, rx, monostatic')
    
    rx(tx[0], tx[1]-resolution, tx[2]-resolution)#, identifier=identifier)

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font="Verdana")
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()