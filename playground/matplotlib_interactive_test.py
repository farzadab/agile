# import matplotlib.pyplot as plt
# import numpy as np
# import ipdb

# x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)
# scale = 200.0 * np.rand(x.shape)

# # You probably won't need this if you're embedding things in a tkinter plot...
# plt.ion()

# cmap = plt.get_cmap('seismic')
# fig,ax = plt.subplots(1,1)
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1)
# line1, = ax.scatter(x, y, cmap=cmap, s=scale,
#                alpha=0.3, edgecolors='none')
# # s=scale, label=color,
# # for phase in np.linspace(0, 10*np.pi, 500):
# #     line1.set_ydata(np.sin(x + phase))
# #     fig.canvas.draw()
# #     fig.canvas.flush_events()


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numpy.random import rand
import numpy as np

plt.ion()

fig, ax = plt.subplots()
# for color in ['red', 'green', 'blue']:
n = 750
d = rand(2, n)
x, y = d
scale = 200.0 * rand(n)
sc = ax.scatter(x, y, norm=Normalize(0,1), c=rand(n)*.1, cmap=plt.get_cmap('seismic'), alpha=0.3, edgecolors='none')

print(sc.get_array())
# ax.legend()
# ax.grid(True)

# plt.show()
# fig.canvas.draw()
# fig.canvas.flush_events()

# import ipdb
# ipdb.set_trace()
for i in range(100):
    
    # print(x,y)
    sc.set_array(rand(n))
    sc.set_offsets(d.transpose())
    fig.canvas.draw()
    fig.canvas.flush_events()