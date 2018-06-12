import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import numpy as np
import time


plt.ion()

class ScatterPlot(object):
    def __init__(self, value_range=[-1,1], xlim=[-1,1], ylim=[-1,1], palette='seismic'):
        cmap = plt.get_cmap(palette)
        norm = matplotlib.colors.Normalize(*value_range)
        # FIXME: ignoring s=scale,
        self.fig, self.ax = plt.subplots(1,1)
        self.sc = self.ax.scatter(x=[], y=[], c=[], norm=norm, cmap=cmap, alpha=0.8, edgecolors='none')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self._redraw()
    
    def _redraw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update(self, points, values):
        # self.sc.set_offsets(np.c_[x,y])
        # assuming points are have the shape (N*2) and consist of x,y coordinates
        self.sc.set_offsets(points)
        self.sc.set_array(values)
        self._redraw()
        

if __name__ == '__main__':
    plot = ScatterPlot()
    for _ in range(100):
        n = 10
        x = np.random.rand(100) * 2 - 1
        y = np.random.rand(100) * 2 - 1
        c = np.random.rand(100)
        plot.update(np.c_[x, y], c)
        time.sleep(0.2)