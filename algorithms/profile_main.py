import profile

from algorithms import main

profile.run('main.main()', 'restats')

import pstats
p = pstats.Stats('restats')
p.strip_dirs().sort_stats('cumtime').print_stats(50)