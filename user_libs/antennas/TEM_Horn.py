# Copyright (C) 2020, Bernd Arendt
# Copyright (C) 2015-2020, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *
from user_libs.antennas.discretGeo import disGeometryGprMax

def horn_burr(x, y, z, resolution = 0.0005, rotation = 0, measurement ='monostatic'):
    """
    Insert a TEM Hornantenna simila to the antenna ..... (insert paperlink)

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna.
        resolution (float): Spatial resolution for the antenna model.
        rotation (float): Rotate model in degrees CCW in xy plane.
        measurement ='monostatic', 'rx', 'tx'
    """
    # Antenna geometry properties
    # z = (C1*exp(R*t)+C2)
    # y = C1_SIDE*exp(R_SIDE*t)+C2_SIDE+(W_FEED/2)-W_FEED/2
    # x = t

    L           = 0.2               # mm
    L_plate     = 0.07
    resolution  = 0.0005            # mm
    R           = 0.019978589752617
    Rside       = 0.011
    B           = 170                   # mm
    tol3d       = 0.0003
    G           = 0.0023 + tol3d    # mm
    arcRadius   = 0.025            # mm
    W1          = 0.13              # mm
    wfeed       = 0.011             # mm
    C1          = ((W1*1e3/2) - (G*1e3/2))/(np.exp(R*L*1e3)-np.exp(R*0))
    C2          = (G*1e3/2 * np.exp(R*L*1e3)-W1*1e3/2 * np.exp(R*0))/(np.exp(R*L*1e3)-np.exp(R*0))  # 0.259141  ( G / 2 mm * exp(R * L / 1 mm) - W1 / 2 mm * exp(0 oE) ) / ( exp(R * L / 1 mm) - exp(0 oE) )
    slopeAngel  = np.arctan(C1*R*np.exp(R*L*1e3))  # 52.291047
    arcAngle    = np.deg2rad(90)                # deg
    arcStart    = (slopeAngel - arcAngle)       # deg
    arcStop     = arcAngle
    arcDelta    = arcStop - arcStart
    Lfeed       = 0.02                  # m
    deltaB      = 0.02                  # m
    B2          = B + deltaB            # m
    lBalun      = 0.014                 # m
    balunDiaBig = 0.011                 # m
    balunDiaSmall = 0.008               # m
    smaDia      = 0.001
    smaTeflonDia= 0.005
    smaSocketDia= 0.008
    smaScrwDia  = 0.006
    sourceresistance = 200              # ohm
    
    # if resolution == 0.001:
    #     dx = 0.001
    #     dy = 0.001
    #     dz = 0.001
    # else:
    #     raise CmdInputError('This antenna module can only be used with a spatial discretisation of 1mm')
    
    material(5, 0, 1, 0, 'smaTeflon')
    material(10, 0, 1, 0, 'myTest')
    material(5, 0, 1, 0, 'myTest2')

    # end of balun/begin horn x, y, z center
    sma_center = x, y, z + lBalun

    ### Balun ###
    # top plate
    box(sma_center[0]+(G/2), sma_center[1]-balunDiaBig/2, z, sma_center[0]+(G/2+resolution), sma_center[1]+balunDiaBig/2, sma_center[2],  'myTest')
    # bottom plate
    box(sma_center[0]-(G/2+resolution), sma_center[1]-balunDiaSmall/2, z, sma_center[0]-(G/2), sma_center[1]+balunDiaSmall/2, sma_center[2],  'myTest')
    # top big part
    cylinder(sma_center[0]+G/2, sma_center[1], sma_center[2], sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], balunDiaBig/2, 'myTest2')
    # substract hole 
    cylinder(sma_center[0]+G/2, sma_center[1], sma_center[2], sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], 0.004/2, 'free_space')
    # bottom small part
    cylinder(sma_center[0]-G/2, sma_center[1], sma_center[2], sma_center[0]-(G/2+resolution), sma_center[1], sma_center[2], balunDiaSmall/2, 'myTest2')


    ### SMA Connector ###
    cylinder(sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.002+resolution, sma_center[1], sma_center[2], smaSocketDia/2, 'pec' )
    cylinder(sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaScrwDia/2, 'pec')
    cylinder(sma_center[0]-(G/2), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaTeflonDia/2, 'smaTeflon')
    cylinder(sma_center[0]-(G/2+resolution+0.001), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaDia/2, 'pec')
    #edge(sma_center[0]-(G/2+0.002), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010, sma_center[1], sma_center[2], 'myTest')

    ### Horn ###
    print('C1:'+str(C1)+' C2:'+str(C2) + ' SlopeAngel:' + str(np.rad2deg(slopeAngel))+'\n'+'ArcStart:'+ str(np.rad2deg(arcStart)))
    z=np.arange(0, (L+resolution)*1e3, resolution*1e3)
    alpha=np.linspace(arcStart, arcStop, num=200, endpoint=True)
    # horn
    x_z=C1 * np.exp((z*R))+C2      # horn part
    y_z=
    # radius
    x_alpah=W1*1e3/2+arcRadius*1e3*np.sin(alpha)+np.cos(slopeAngel)*arcRadius*1e3
    z_alpha=L*1e3+arcRadius*1e3*np.cos(alpha)-np.sin(slopeAngel)*arcRadius*1e3

    zz, xx_z = disGeometryGprMax(z, x_z, resolution*1e3)
    z_a, x_a = disGeometryGprMax(z_alpha,x_alpah, resolution*1e3)

    plt.plot(zz, xx_z, z_a, x_a)
    plt.axis([0, 220, 0, 120])
    plt.show()

    ### Source on SMA Connector ###
    tx = sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2]
    print('#waveform: gaussian 1 1e9 myGaussian')
    voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution))

    ### Reciever on SMA Connector ###
    if measurement == 'monostatic':
        rx = tx
        identifier = 'monostatic'
    elif measurement == 'rx':
        identifier = 'rxhorn'
    elif measurement == 'tx':
        identifier = 'txhorn'
    else:
        raise CmdInputError('This antenna have 3 measuremnt methods - tx, rx, monostatic')
    
    #rx(tx[0] - 0.059, tx[1], tx[2], identifier=identifier, polarisation='y', dxdy=(resolution, resolution))
