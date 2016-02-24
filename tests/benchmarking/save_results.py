import os, platform
import numpy as np

moduledirectory = os.path.dirname(os.path.abspath(__file__))

# Machine identifier
platformID = platform.platform()
machineID = 'MacPro1,1'
machineIDlong = machineID + ' (2006); 2 x 2.66 GHz Quad-Core Intel Xeon; Mac OS X 10.11.3'
#machineID = 'iMac15,1'
#machineIDlong = machineID + ' (Retina 5K, 27-inch, Late 2014); 4GHz Intel Core i7; Mac OS X 10.11.3'

# Number of threads (0 signifies serial compiled code)
threads = np.array([1, 2, 4, 8])

# 100 x 100 x 100 cell model execution times (seconds)
bench1 = np.array([149, 115, 100, 107])

# 150 x 150 x 150 cell model execution times (seconds)
bench2 = np.array([393, 289, 243, 235])

# Save to file
np.savez(os.path.join(moduledirectory, machineID), threads=threads, bench1=bench1, bench2=bench2)


