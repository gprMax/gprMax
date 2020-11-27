import numpy as np
import matplotlib.pyplot as plt

L           = 0.2           # mm
L_plate     = 0.07
resolution  = 0.0005        # mm
R           = 0.019978589752617
G           = 0.0024        # mm
slopeAngel  = 52.291047
arcAngle    = np.deg2rad(90)            # deg
arcStart    = np.deg2rad((slopeAngel - 90)) #deg
arcStop     = arcAngle
arcDelta    = arcStop - arcStart
arcRadius   = 0.025     # mm
W1          = 0.13          # mm
C1          = ((W1*1e3/2) - (G*1e3/2))/(np.exp(R*L*1e3)-np.exp(R))                          #1.190859  ( W1 / ( 2 mm ) - G / ( 2 mm ) ) / ( exp(R * L / 1 mm) - exp(R * 0 oE) )
C2          = (G*1e3/2 * np.exp(R*L*1e3)-W1*1e3/2 * np.exp(0))/(np.exp(R*L*1e3)-np.exp(0))  #0.259141  ( G / 2 mm * exp(R * L / 1 mm) - W1 / 2 mm * exp(0 oE) ) / ( exp(R * L / 1 mm) - exp(0 oE) )

print('C1:'+str(C1)+'C2:'+str(C2))
### geometry ###
l       = np.arange(0, (L+resolution)*1e3, resolution*1e3)
alpha   = np.linspace(arcStart, arcStop, num = 200, endpoint=True)   #SLOPE_ANGLE / 1 grd - 90 oE ARC_ANGLE / 1 grd
#ll      = np.arange(L_plate*1e3, (L+resolution)*1e3, resolution*1e3)
print(np.rad2deg(alpha))
# horn
x_l     = C1 * np.exp((l*R))+C2      # horn part

# radius 
# Z:ARC_RADIUS*sin(t)+W1/2+cos(SLOPE_ANGLE)*ARC_RADIUS
# X:ARC_RADIUS*cos(t)+L-sin(SLOPE_ANGLE)*ARC_RADIUS
y_alpah = W1/2*1e3+arcRadius*1e3*np.sin(alpha)-np.cos(slopeAngel)*arcRadius*1e3
x_alpha = L*1e3+arcRadius*1e3*np.cos(alpha)-np.sin(slopeAngel)*arcRadius*1e3

print(y_alpah,x_alpha)

plt.plot(l, x_l, x_alpha, y_alpah)
plt.axis([0, 220, 0, 120])
plt.show()