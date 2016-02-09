"""gprMax.__main__: executed when gprMax directory is called as script."""

import gprMax.gprMax

if __name__ == '__main__':
    gprMax.gprMax.main()

# Code profiling
# Time profiling
#import cProfile, pstats
#cProfile.run('main()','stats')
#p = pstats.Stats('stats')
#p.sort_stats('time').print_stats(50)

# Memory profiling - use in gprMax.py
# from memory profiler import profile
# add @profile before function to profile
