import random as rd
from gprMax.input_cmd_funcs import *

def gravel(dSmin, dSmax, dxSmin, dxSmax, surfaceS, simx, simy, simz, material):
    """
    min/max Stone diameter, min max Stone distance, max domain
    """
    intFac = 1000
    dSmin       = int(dSmin *intFac)        ## pebble diameter min
    dSmax       = int(dSmax *intFac)		## pebble diameter max
    dxSmin      = int(dxSmin*intFac)		## distance center x
    dxSmax      = int(dxSmax*intFac)		## distance center y
    
    ##Pebble arrangement
    yy = dSmax/2/intFac
    while yy <= surfaceS-(dSmin/2/intFac):    
        xx =  (rd.randrange(70, 130, 10)/100)*dSmax/2/intFac
        while xx <= simx-(dSmax/2/intFac):
            dS = rd.randrange(dSmin, dSmax, 2)/intFac
            dy = rd.randrange(dxSmin, dSmax, 2)/intFac
            y = yy + dy
            dx = rd.randrange(dxSmin, dSmax, 2)/intFac
            x = xx + dx
            cylinder(x, y, 0, x, y, simz, dS/2 ,material)
            xx = xx + dx
        yy = yy+dxSmax/2/intFac