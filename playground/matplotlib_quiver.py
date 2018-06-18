import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import time

X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U = np.cos(X)
V = np.sin(Y)

plt.ion()
fig, ax = plt.subplots()
M = np.hypot(U, V)
Q = ax.quiver(X, Y, U, V, M, units='xy', pivot='middle', scale=1 / 0.15)

fig.canvas.draw()
fig.canvas.flush_events()

import ipdb
ipdb.set_trace()

const = 10
for i in range(1, 20):
    time.sleep(.1)
    Q.set_UVC(U + i/2/const, V, M*const/(i+const))
    fig.canvas.draw()
    fig.canvas.flush_events()
