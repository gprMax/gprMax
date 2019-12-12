"""gprMax.__main__: executed when gprMax directory is called as script."""

import gprMax.cli

if __name__ == '__main__':
    gprMax.cli.main()

# Code profiling
# Time profiling
# import cProfile, pstats
# cProfile.run('gprMax.gprMax.main()','stats')
# p = pstats.Stats('stats')
# p.sort_stats('time').print_stats(25)

# Memory profiling - use in cli.py
# from memory profiler import profile
# add @profile before function to profile
