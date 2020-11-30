import numpy as np
import matplotlib.pyplot as plt

def disGeometryGprMax(x, y, resolution):
    """
    deiscret geometryfunction in different parts
    """
    i=0
    ys = np.array([0], dtype=float)
    xn = np.array([0], dtype=float)

    for n in range(0, len(y)):
        if n == 0:
            ys[0] = resolution*round(y[0]/resolution)
            xn[0] = resolution*round(x[0]/resolution)
            #print(ys[-1])
        
        if (y[n])-ys[-1] >=resolution:            
            #print(x[n],y[n])
            i = i+1            
            #print('deltaX:'+str(x[n]-xn[-1])+'\n'+'deltaY: '+str(y[n]-ys[-1]))
            fitX = round((x[n]-xn[-1])/resolution)*resolution
            fitY = round((y[n]-ys[-1])/resolution)*resolution
            #print(str(fitX)+'fitDeltaY:'+str(fitY))
            xn = np.append(xn, xn[-1]+fitX)
            ys = np.append(ys, ys[-1]+fitY)            
    
    # check last point
    dx = x[-1]-xn[-1]
    if np.abs(dx) >= resolution:
        fitX = round((x[-1]-xn[-1])/resolution)*resolution
        fitY = round((y[-1]-ys[-1])/resolution)*resolution
        xn = np.append(xn, xn[-1]+fitX)
        ys = np.append(ys, ys[-1]+fitY)    

    # plt.plot(xn, ys)
    # plt.axis([0, 220, 0, 120])
    # plt.show()

    return xn, ys

def createPoints(self, parameter_list):
    """
    docstring
    """
    pass

L           = 0.2               # m
L_plate     = 0.07
resolution  = 0.0005            # m
R           = 0.019978589752617
Rside       = 0.011
B           = 0.17                   # m
tol3d       = 0.0003
G           = 0.0023 + tol3d    # m
arcRadius   = 0.025            # m
W1          = 0.13              # m
wfeed       = 0.011 + tol3d          # m
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
balunDiaBig = 0.011                 # m
balunDiaSmall = 0.008               # m
smaDia      = 0.001
smaTeflonDia= 0.005
smaSocketDia= 0.008
smaScrwDia  = 0.006
sourceresistance = 200              # ohm

print('C1:'+str(C1)+' C1side:'+str(C1_side)+' C2:'+str(C2)+' C2side:'+str(C2_side) + ' SlopeAngel:' + str(np.rad2deg(slopeAngel))+'\n'+'ArcStart:'+ str(np.rad2deg(arcStart)))
zl = np.arange(0, (L+resolution)*1e3, resolution*1e3)
alpha = np.linspace(arcStart, arcStop, num=200, endpoint=True)
# horn
x_z = C1 * np.exp((zl*R))+C2      # horn part
y_z = C1_side*np.exp(Rside*zl)+C2_side+(wfeed*1e3/2)-wfeed*1e3/2   # c1_side*efk^(R_side*v*u)+c2_side+(w_feed/2*v)-w_feed/2 # C1_SIDE*exp(R_SIDE*t)+C2_SIDE+(W_FEED/2mm)-W_FEED/2
x_alpah = W1*1e3/2+arcRadius*1e3*np.sin(alpha)+np.cos(slopeAngel)*arcRadius*1e3
y_alpah = B*1e3/2+deltaB*1e3/2*(alpha-arcStart)/arcDelta #B/2+(DELTA_B)/2*(t-ARC_START/1grd)/(ARC_DELTA/1grd)
z_alpha = L*1e3+arcRadius*1e3*np.cos(alpha)-np.sin(slopeAngel)*arcRadius*1e3

# discret funcion
zzy, yy_z = disGeometryGprMax(zl, y_z, resolution*1e3)
zzx, xx_z = disGeometryGprMax(zl, x_z, resolution*1e3)
zx_a, x_za = disGeometryGprMax(z_alpha, x_alpah, resolution*1e3)
zy_a, y_za = disGeometryGprMax(z_alpha, y_alpah, resolution*1e3)

# build geometry

zz = np.append(zzy, zzx)
zz = np.sort(zz)
print('zzy:'+str(zzy)+'\n zzx'+str(zzx)+'\n zconnect and sorted:'+str(np.unique(zz)))

# zreal = z-zzx/1e3
# xx_zreal = x + xx_z/1e3
plt.plot(zzx, xx_z, zx_a, x_za, zzy, yy_z, zy_a, y_za)
plt.axis('equal')
#plt.axis([0, 220, 0, 120])
# plt.plot(zreal, xx_zreal)
plt.show()