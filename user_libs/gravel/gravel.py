import random as rd
from gprMax.input_cmd_funcs import *

def gravel(dSmin, dSmax, dxSmin, dxSmax, dySmin, dySmax, surfaceS, simx, simy, simz, material):
    """
    min/max Stone diameter, Faktor min max Stone distance (z.B. 70 130), max domain
    """
    intFac = 1000
    dSmin       = int(dSmin *intFac)        ## pebble diameter min
    dSmax       = int(dSmax *intFac)		## pebble diameter max
    ##dxSmin      = int(dxSmin*intFac)		## distance center x
    ##dxSmax      = int(dxSmax*intFac)		## distance center y
    
    ##Pebble arrangement
    yy = 0.4 * dSmax/2/intFac
    while yy <= surfaceS-(dSmin/2/intFac):    
        xx =  (rd.randrange(70, 130, 10)/100)*dSmax/2/intFac
        while xx <= simx-((dxSmax/2+dSmax)/intFac)*0.7:
            dS = rd.randrange(dSmin, dSmax, 2)/intFac
            ##dy = rd.randrange(dxSmin, dSmax, 2)/intFac
            ##dx = rd.randrange(dxSmin, dSmax, 2)/intFac
            dy = dS * rd.randrange(dySmin, dySmax, 10)/100
            dx = dS * rd.randrange(dxSmin, dxSmax, 10)/100
            y = yy + dy
            x = xx + dx
            cylinder(x, y, 0, x, y, simz, dS/2 ,material)
            xx = x
        yy = yy + dS * rd.randrange(dySmin, dySmax, 10)/100 
        ##yy = yy+dxSmax/2/intFac


def gravel2(dSmin, dSmax, dxSmin, dxSmax, surfaceS, simx, simy, simz, material):
    """
    min/max Stone diameter, min max Stone distance (z.B. 0.005), max domain
    """
    intFac = 1000
    dSmin       = int(dSmin *intFac)        ## pebble diameter min
    dSmax       = int(dSmax *intFac)		## pebble diameter max
    dxSmin      = int(dxSmin*intFac)		## distance center x
    dxSmax      = int(dxSmax*intFac)		## distance center y
    
    
    ##Pebble arrangement
    yy = 0.4 * dSmax/2/intFac
    while yy <= surfaceS-(dSmin/2/intFac):    
        xx =  (rd.randrange(30, 50, 10)/100)*dSmax/2/intFac
        while xx <= simx-((dxSmax/2+dSmax)/intFac)*0.7:
            dS = rd.randrange(dSmin, dSmax, 2)/intFac
            dy = rd.randrange(dxSmin, dSmax, 2)/intFac
            dx = rd.randrange(dxSmin, dSmax, 2)/intFac
            y = yy + dy
            x = xx + dx
            cylinder(x, y, 0, x, y, simz, dS/2 ,material)
            xx = x
        yy = yy+dxSmax/2/intFac