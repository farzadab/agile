import os

import numpy as np

from bokeh.plotting import curdoc, figure, show, output_file
from bokeh.palettes import magma, inferno, viridis
from bokeh.models import ColumnDataSource
from bokeh.models.mappers import LinearColorMapper
import ipdb

# os.environ['BOKEH_LOG_LEVEL'] = 'error'

# class ScatterPlot(object):
#     def __init__(self, low=-1, high=1, palette=inferno(50)):
#         self.cmap = seismic
#         self.color_mapper = LinearColorMapper(low=low, high=high, palette=palette)
#         self.source = ColumnDataSource(dict(x=[], y=[], value=[]))
    
#     def update(self, points):
#         self.source.update


# def scatter_plot( ):

# N = 4000
# x = np.random.random(size=N) * 100
# y = np.random.random(size=N) * 100
# radii = np.random.random(size=N) * 1.5
# # colors = [
# #     "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
# # ]

# source = ColumnDataSource(dict(x=x,y=y,radius=radii))

# # ipdb.set_trace()

# TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

# p = figure(tools=TOOLS)
# doc = curdoc()

# mapper = LinearColorMapper(palette=inferno(50),low=0,high=100)
# p.scatter(x='x', y='y', radius='radius', source=source,
#           fill_color={'field': 'x', 'transform': mapper}, fill_alpha=0.6,
#           line_color=None)

# # output_file("color_scatter.html", title="color_scatter.py example")

# # show(p)  # open a browser
# doc.add_root(p)

# import time

# for i in range(100):
#     time.sleep(1)
#     radii = np.random.random(size=N) * 1.5
#     source.update(data=dict(x=x,y=y,radius=radii))
#     # ipdb.set_trace()
#     # source.update(data=np.random.random(size=N) * 100)


# # p1 = figure(width=1500, height=230, active_scroll="wheel_zoom")


# # p2 = figure(width=1500, height=500, active_scroll="wheel_zoom")

# # layout.children[0] = p2