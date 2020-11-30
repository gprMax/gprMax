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

# L = 0.2*1e3                  # mm
# L_plate = 0.07*1e3
# resolution = 0.0005*1e3            # mm
# R = 0.019978589752617
# Rside = 0.011
# B = 170 # mm
# tol3d = 0.0003*1e3
# G = 0.0023*1e3  +tol3d      # mm
# arcRadius = 0.025*1e3         # mm
# W1 = 0.13*1e3          # mm
# wFeed = 0.011*1e3   # mm
# # 1.190859  ( W1 / ( 2 mm ) - G / ( 2 mm ) ) / ( exp(R * L / 1 mm) - exp(R * 0 oE) )
# C1 = ((W1/2) - (G/2))/(np.exp(R*L)-np.exp(R*0))
# C2 = (G/2 * np.exp(R*L)-W1/2 * np.exp(R*0))/(np.exp(R*L)-np.exp(R*0))  # 0.259141  ( G / 2 mm * exp(R * L / 1 mm) - W1 / 2 mm * exp(0 oE) ) / ( exp(R * L / 1 mm) - exp(0 oE) )
# slopeAngel=np.arctan(C1*R*np.exp(R*L))  # 52.291047
# arcAngle=np.deg2rad(90)                # deg
# arcStart=(slopeAngel - arcAngle)  # deg
# arcStop=arcAngle
# arcDelta=arcStop - arcStart

# print('C1:'+str(C1)+' C2:'+str(C2) + ' SlopeAngel:' + str(np.rad2deg(slopeAngel))+'\n'+'ArcStart:'+ str(np.rad2deg(arcStart)))
# ### geometry ###
# l=np.arange(0, (L+resolution), resolution)
# # SLOPE_ANGLE / 1 grd - 90 oE ARC_ANGLE / 1 grd
# alpha=np.linspace(arcStart, arcStop, num=200, endpoint=True)
# # ll      = np.arange(L_plate*1e3, (L+resolution)*1e3, resolution*1e3)
# # horn
# x_l=C1 * np.exp((l*R))+C2      # horn part

# # radius
# # Z:ARC_RADIUS*sin(t)+W1/2+cos(SLOPE_ANGLE)*ARC_RADIUS
# # X:ARC_RADIUS*cos(t)+L-sin(SLOPE_ANGLE)*ARC_RADIUS
# y_alpah=W1/2+arcRadius*np.sin(alpha)+np.cos(slopeAngel)*arcRadius
# x_alpha=L+arcRadius*np.cos(alpha)-np.sin(slopeAngel)*arcRadius

# xx, yy = disGeometryGprMax(l, x_l, resolution)
# print('x: '+str(xx)+'\n'+'y: '+str(yy))
# print(len(xx), len(yy))

# xa, ya = disGeometryGprMax(x_alpha, y_alpah, resolution)

# plt.plot(l, x_l, x_alpha, y_alpah, xx, yy, xa, ya)
# plt.axis([0, 220, 0, 120])
# plt.show()

