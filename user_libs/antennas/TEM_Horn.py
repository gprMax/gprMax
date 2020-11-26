# Copyright (C) 2020, Bernd Arendt
# Copyright (C) 2015-2020, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506
import numpy as np

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *

def horn_burr(x, y, z, resolution = 0.001, rotation = 0, measurement ='monostatic'):
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
    L       = 0.2           # mm
    G       = 0.0024        # mm
    wfeed   =  0.011        # mm
    R       = 0.019978589752617
    B       = 0.17          # mm
    Lfeed   = 0.02          # mm
    W1      = 0.13          # mm
    #C1      = W1 /(2) - G/(2) / (np.exp(R*L) - np.exp(R)
    C1      = 1.190859
    deltaB  = 0.02          # mm
    B2      = B + deltaB    # mm
    slopeAngel  = 52.291
    lBalun  = 0.014         # mm
    balunDiaBig = 0.011     # mm
    balunDiaSmall = 0.008    # mm
    #c1Foil  = ( W_FOIL / 2 mm - ( G / 2 mm + D_FOIL / 1 mm ) ) / ( exp(R * L_FOIL_GES / 1 mm) - exp(R * 0 oE) ) = 1.191475
    c1Foil  = 1.191475
    C2      = 0.259141
    C1side  = 9.906525
    C2side  = -4.406525
    arcRadius   = 0.025     # mm
    arcAngle    = 90        # deg
    arcStart    = slopeAngel - 90 #deg
    arcStop     = arcAngle
    arcDelta    = arcStop - arcStart
    smaDia      = 0.001
    smaTeflonDia   = 0.005
    smaSocketDia    = 0.008
    smaScrwDia  = 0.006
    sourceresistance = 200 # ohm
    
    # if resolution == 0.001:
    #     dx = 0.001
    #     dy = 0.001
    #     dz = 0.001
    # else:
    #     raise CmdInputError('This antenna module can only be used with a spatial discretisation of 1mm')
    
    material(5, 0, 1, 0, 'smaTeflon')
    material(10, 0, 1, 0, 'myTest')

    # end of balun/begin horn x, y, z center
    sma_center = x, y, z + lBalun

    ### Balun ###
    # top plate
    box(sma_center[0]+(G/2), sma_center[1]-balunDiaBig/2, z, sma_center[0]+(G/2+resolution), sma_center[1]+balunDiaBig/2, sma_center[2],  'myTest')
    # bottom plate
    box(sma_center[0]-(G/2+resolution), sma_center[1]-balunDiaSmall/2, z, sma_center[0]-(G/2), sma_center[1]+balunDiaSmall/2, sma_center[2],  'myTest')
    # top big part
    cylinder(sma_center[0]+G/2, sma_center[1], sma_center[2], sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], balunDiaBig/2, 'pec')
    # substract hole 
    cylinder(sma_center[0]+G/2, sma_center[1], sma_center[2], sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], 0.004/2, 'free_space')
    # bottom small part
    cylinder(sma_center[0]-G/2, sma_center[1], sma_center[2], sma_center[0]-(G/2+resolution), sma_center[1], sma_center[2], balunDiaSmall/2, 'pec')


    ### SMA Connector ###
    cylinder(sma_center[0]+(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.002+resolution, sma_center[1], sma_center[2], smaSocketDia/2, 'pec' )
    cylinder(sma_center[0]-(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaScrwDia/2, 'pec')
    cylinder(sma_center[0]-(G/2+resolution), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaTeflonDia/2, 'smaTeflon')
    cylinder(sma_center[0]-(G/2+resolution+0.002), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010+resolution, sma_center[1], sma_center[2], smaDia/2, 'pec')
    #edge(sma_center[0]-(G/2+0.002), sma_center[1], sma_center[2], sma_center[0]+G/2+0.010, sma_center[1], sma_center[2], 'myTest')

    ### Source on SMA Connector ###
    tx = sma_center[0]+G/2+0.010, sma_center[1], sma_center[2]
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
    
    #rx(tx[0] - 0.059, tx[1], tx[2], identifier=identifier, to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

  
    
