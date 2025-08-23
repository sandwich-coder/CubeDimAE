import sys, os, subprocess

from copy import deepcopy as copy
import inspect, code
import types
import time
import numpy as np
from scipy import linalg as la
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams.update({
    'figure.figsize':[10, 10],
    'figure.edgecolor':'none',
    'figure.facecolor':'none',
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.edgecolor':'none',
    'axes.facecolor':'none',
    'lines.markersize':1,
    'lines.linewidth':0.5,
    'legend.edgecolor':'none',
    'legend.facecolor':'none',
    })

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve, make_swiss_roll

#directories
os.makedirs('figures', exist_ok = True)

#preparatory functions
def lims(data):
    ranges = data.max(axis = 0) - data.min(axis = 0)
    width = ranges.max().tolist()
    width = width * 1.2    #margins
    box = np.array(
        [[0] * 3, [width] * 3],
        dtype = 'float64',
        )
    
    middle_box = (box.max(axis = 0) + box.min(axis = 0)) / np.float64(2)
    middle_data = (data.max(axis = 0) + data.min(axis = 0)) / np.float64(2)
    move = middle_data - middle_box
    limits = box + move
    
    return limits.transpose().tolist()


# - moons -

moons = make_moons(n_samples = 40, random_state = 1)[0]

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(1)
ax.set_aspect(2)
ax.set_xticks([])
ax.set_yticks([])
plot = ax.plot(
    moons[:, 0], moons[:, 1],
    marker = 'o', markersize = 10,
    linestyle = '',
    )
fig.savefig('figures/moons_.png', dpi = 600)

del fig, ax, plot


# - datasets -

datasets = {}

# s curve
datasets['s_curve'] = make_s_curve(n_samples = 3200, random_state = 1)[0]

# swiss roll
datasets['swiss_roll'] = make_swiss_roll(n_samples = 3200, random_state = 1)[0]

# mobius strip
theta = np.linspace(0, 2 * np.pi, num = 200, endpoint = False, dtype = 'float64')
w = np.linspace(-0.25, 0.25, 16)
w, theta = np.meshgrid(w, theta)
phi = -0.5 * theta    # multiple of 0.5
r = 1 + w * np.cos(phi)
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))
datasets['mobius_strip'] = np.stack([x, y, z], axis = 1)
del theta, w, phi, r, x, y, z

#hollow sphere
radius = np.array([1], dtype = 'float64')
azimuth = np.flip(np.linspace(np.pi, -np.pi, num = 80, endpoint = False, dtype = 'float64'))
elevation = np.linspace(-np.pi / 2, np.pi / 2, num = 40, dtype = 'float64')
radius, azimuth, elevation = np.meshgrid(radius, azimuth, elevation, copy = False)
radius = radius.ravel()
azimuth = azimuth.ravel()
elevation = elevation.ravel()
x1 = radius * np.cos(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x2 = radius * np.sin(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x3 = radius * np.sin(elevation, dtype = 'float64')
datasets['hollow_sphere'] = np.stack([x1, x2, x3], axis = 1)
del radius, azimuth, elevation, x1, x2, x3

#solid sphere
displacement = np.random.uniform(low = 0, high = 1, size = [3200]) ** np.float64(1 / 3)
azimuth = np.random.uniform(low = -np.pi, high = np.pi, size = [3200])
elevation = np.random.uniform(low = -1, high = 1, size = [3200])
elevation = np.arcsin(elevation, dtype = 'float64')
x1 = displacement * np.cos(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x2 = displacement * np.sin(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x3 = displacement * np.sin(elevation, dtype = 'float64')
datasets['solid_sphere'] = np.stack([x1, x2, x3], axis = 1)
del displacement, azimuth, elevation, x1, x2, x3

for l in datasets.keys():
    xlim, ylim, zlim = lims(datasets[l])
    fig = pp.figure(layout = 'constrained')
    ax = fig.add_subplot(projection = '3d')
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.view_init(azim = 75, elev = 20)

    ax.xaxis.pane.set(color = 'none')
    ax.yaxis.pane.set(color = 'none')
    ax.zaxis.pane.set(color = 'none')
    
    plot = ax.scatter(
        datasets[l][:, 0],
        datasets[l][:, 1],
        datasets[l][:, 2],
        alpha = 0.5,
        )
    fig.savefig('figures/{name}'.format(name = l), dpi = 600)
    
del xlim, ylim, zlim, fig, ax, plot
