# -- Experiment Script --

import os
import time
import logging
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(name = 'experiment')

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['figure.figsize'] = (6, 6)
mpl.use('Agg')

import tensorflow as tf
from tensorflow.keras import layers

import yaml
from sklearn.datasets import make_s_curve, make_swiss_roll

from cube_dim import CubeDim

# - initialized -

config = yaml.load(
    open('config.yml', 'r'),
    Loader = yaml.FullLoader,
    )

#functions
def translate3d(points, move_x = 0, move_y = 0, move_z = 0):
    translation = np.array([move_x, move_y, move_z], dtype = 'float64')
    return points + translation.reshape([-1, translation.shape[0]])
def rotate3d(points, angle_x = 0, angle_y = 0, angle_z = 0):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle_x, dtype = 'float64'), -np.sin(angle_x, dtype = 'float64')],
                           [0, np.sin(angle_x, dtype = 'float64'), np.cos(angle_x, dtype = 'float64')]],
                          dtype = 'float64')
    rotation_y = np.array([[np.cos(angle_y, dtype = 'float64'), 0, np.sin(angle_y, dtype = 'float64')],
                           [0, 1, 0],
                           [-np.sin(angle_y, dtype = 'float64'), 0, np.cos(angle_y, dtype = 'float64')]],
                          dtype = 'float64')
    rotation_z = np.array([[np.cos(angle_z, dtype = 'float64'), -np.sin(angle_z, dtype = 'float64'), 0],
                           [np.sin(angle_z, dtype = 'float64'), np.cos(angle_z, dtype = 'float64'), 0],
                           [0, 0, 1]],
                          dtype = 'float64')
    return points @ rotation_x.transpose() @ rotation_y.transpose() @ rotation_z.transpose()
def lims(A):
    ranges = A.max(axis = 0) - A.min(axis = 0)
    width = ranges.max().tolist()
    width = width * 1.2    #margins
    box = np.array([[0] * 3,
                    [width] * 3], dtype = 'float64')

    middle_box = (box.max(axis = 0) + box.min(axis = 0)) / np.float64(2)
    middle_A = (A.max(axis = 0) + A.min(axis = 0)) / np.float64(2)
    move = middle_A - middle_box
    limits = box + move
    return limits.transpose().tolist()

def voxel(data, length, return_colors = False, return_limits = True):
    nodes = np.round(data / length)
    nodes = nodes.astype('int64', copy = False)
    nodes, counts = np.unique(nodes, axis = 0, return_counts = True)
    density = counts.astype('float64', copy = False) / counts.max().astype('float64', copy = False)
    density = density.max() - density    #inversion
    density = density * np.float64(0.7) + np.float64(1 - 0.7)    #brightening
    gridsize = nodes.max(axis = 0) - nodes.min(axis = 0) + np.int64(1)
    gridsize = gridsize.tolist()
    voxels = np.full(gridsize, False)
    switches = nodes - nodes.min(axis = 0, keepdims = True)
    switch = switches.tolist()
    ret = []
    for ll in switch:
        voxels[*ll] = True
    ret.append(voxels)
    if return_colors == True:

        rgb = [0, 1, 0]    #rgb ratio

        rgb = np.array(rgb, dtype = 'float64')
        colors = np.stack([voxels] * 3, axis = 3)
        colors = colors.astype('float64', copy = False)
        for lll in range(len(switch)):
            colors[*switch[lll]] = rgb * density[lll]
        ret.append(colors)
    if return_limits == True:

        width = switches.max() + np.int64(1)
        width = width.tolist()
        width = width + 4    #margins
        low = np.zeros([3], dtype = 'int64')
        high = np.array([width] * 3, dtype = 'int64')
        limits = np.stack([low, high], axis = 0)

        middle_box = high.astype('float64', copy = False) / np.float64(2)
        middle_switches = switches.max(axis = 0).astype('float64', copy = False) / np.float64(2)
        move = middle_switches - middle_box
        limits = limits.astype('float64', copy = False) + move

        limits = limits.transpose().tolist()
        ret.append(limits)

    return ret


#directories
os.makedirs('figures', exist_ok = True)

#estimator
estimator = CubeDim()

#autoencoder
class Autoencoder(tf.keras.Model):
    def __init__(self, code_dim):
        if not isinstance(code_dim, int):
            raise TypeError('The compression dimension should be an integer.')
        if code_dim < 1:
            raise ValueError('The compression dimension should be greater than 0.')
        super().__init__()

        self.encoder = tf.keras.Sequential([

            layers.Dense(100, activation = 'gelu'),
            layers.Dense(99, activation = 'gelu'),
            layers.Dense(98, activation = 'gelu'),
            layers.Dense(97, activation = 'gelu'),
            layers.Dense(96, activation = 'gelu'),

            layers.Dense(code_dim, activation = 'gelu')

            ])

        self.decoder = tf.keras.Sequential([

            layers.Dense(96, activation = 'gelu'),
            layers.Dense(97, activation = 'gelu'),
            layers.Dense(98, activation = 'gelu'),
            layers.Dense(99, activation = 'gelu'),
            layers.Dense(100, activation = 'gelu'),

            layers.Dense(3, activation = 'sigmoid')

            ])


    def call(self, data):
        if not tf.is_tensor(data):
            raise TypeError('The dataset is not a \'tensorflow.Tensor\'.')
        if data.dtype != tf.float32:
            raise ValueError('The dataset is not of \'tensorflow.float32\'.')

        mins = tf.math.reduce_min(data, axis = 0, keepdims = True)
        maxs = tf.math.reduce_max(data, axis = 0, keepdims = True)

        #scaled
        scaled = (data - mins) / (maxs - mins)

        #train
        encoded = self.encoder(scaled)
        decoded = self.decoder(encoded)

        #unscaled
        unscaled = decoded * (maxs - mins) + mins

        return unscaled




# - datasets -

datasets = {}
for l in config['datasets'].keys():
    if config['datasets'][l]:
        datasets[l] = {}

for l in datasets.keys():
    datasets[l]['lapses'] = {}

seed = None

#s curve (2-dimensional)
s_curve = make_s_curve(n_samples = 3200, random_state = seed)[0]

#swiss roll (2-dimensional)
swiss_roll = make_swiss_roll(n_samples = 3200, random_state = seed)[0]

#mobius strip (2-dimensional)
theta = np.linspace(0, 2 * np.pi, num = 200, endpoint = False, dtype = 'float64')
w = np.linspace(-0.25, 0.25, num = 16, dtype = 'float64')
w, theta = np.meshgrid(w, theta)
phi = np.float64(-0.5) * theta
r = np.float64(1) + w * np.cos(phi)
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))
mobius_strip = np.stack([x, y, z], axis = 1)
del theta, w, phi, r, x, y, z

#hollow sphere (2-dimensional)
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
hollow_sphere = np.stack([x1, x2, x3], axis = 1)
del radius, azimuth, elevation, x1, x2, x3

#solid sphere (3-dimensional)
rng = np.random.default_rng(seed = seed)
displacement = rng.uniform(low = 0, high = 1, size = [3200]) ** np.float64(1 / 3)
azimuth = rng.uniform(low = -np.pi, high = np.pi, size = [3200])
elevation = rng.uniform(low = -1, high = 1, size = [3200])
elevation = np.arcsin(elevation, dtype = 'float64')
x1 = displacement * np.cos(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x2 = displacement * np.sin(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x3 = displacement * np.sin(elevation, dtype = 'float64')
solid_sphere = np.stack([x1, x2, x3], axis = 1)
del rng, displacement, azimuth, elevation, x1, x2, x3

#stored
for l in datasets.keys():
    exec(f'datasets[\'{l}\'][\'array\'] = {l}')

for l in datasets.keys():
    xlim, ylim, zlim = lims(datasets[l]['array'])
    fig = plt.figure(layout = 'constrained')
    ax = fig.add_subplot(projection = '3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(azim = 75, elev = 20)
    plot = ax.scatter(
        datasets[l]['array'][:, 0],
        datasets[l]['array'][:, 1],
        datasets[l]['array'][:, 2],
        alpha = 0.3,
        )
    datasets[l]['figure'] = fig    #stored
del xlim, ylim, zlim, fig, ax, plot

#saved
for l in datasets.keys():
    datasets[l]['figure'].savefig(f'figures/{l}_data.png', dpi = 300)

del seed


# - estimation -

for l in datasets.keys():

    begin = time.time()
    estimated = estimator.estimate(datasets[l]['array'])
    end = time.time()
    lapse = end - begin
    logger.info(f'Estimated: {estimated}')
    logger.info(f'    Lapse: {lapse:.2f} (s)')

    #stored
    datasets[l]['estimated'] = estimated
    datasets[l]['lapses']['estimation'] = lapse

del begin, estimated, end, lapse


# - training -

bottlenecks = [1, 2, 3, 4, 5]
epochs = 50
batch_size = 32
colors = {
    1:'red',
    2:'green',
    3:'burlywood',
    4:'cyan',
    5:'pink',
    }

for l in datasets.keys():

    fig = plt.figure(layout = 'constrained')
    gs = fig.add_gridspec(nrows = 2, ncols = 1)
    gs_2 = gs[-1+2].subgridspec(nrows = 1, ncols = 2)

    #bottlenecks cycled
    training_losses = {}
    for ll in bottlenecks:

        data = tf.constant(datasets[l]['array'], dtype = 'float32')

        autoencoder = Autoencoder(ll)
        autoencoder.compile(optimizer = 'adam', loss = 'mse')

        logger.info(f'{l} with bottleneck {ll}')
        begin = time.time()
        history = autoencoder.fit(
            data, data,
            batch_size = 32,
            epochs = 50,
            shuffle = True,
            )
        end = time.time()
        lapse = end - begin
        logger.info(f'elapsed time: {lapse:.2f} (s)')

        training_losses[ll] = history.history['loss']

        reconstructed = autoencoder.predict(data)
        reconstructed = reconstructed.astype('float64')

        #stored
        datasets[l]['lapses'][f'AE{ll}'] = lapse
        datasets[l][f'reconstructed{ll}'] = reconstructed


    ax_1 = fig.add_subplot(gs[-1+1])
    ax_1.set_box_aspect(0.5)
    ax_1.set_title('Training Loss')
    ax_1.set_xlabel('epoch')
    ax_1.set_ylabel('loss')
    plots_1 = []
    for ll in bottlenecks:
        plots_1.append(ax_1.plot(
            np.arange(epochs)+1, training_losses[ll],
            color = colors[ll],
            label = f'{ll}-dimensional',
            ))
    ax_1.legend()

    xlim, ylim, zlim = lims(datasets[l]['array'])

    #1-dimensional compression
    ax_2 = fig.add_subplot(gs_2[-1+1], projection = '3d')
    ax_2.set_box_aspect([1, 1, 1])
    ax_2.set_xlim(*xlim)
    ax_2.set_ylim(*ylim)
    ax_2.set_zlim(*zlim)
    ax_2.set_xticklabels([])
    ax_2.set_yticklabels([])
    ax_2.set_zticklabels([])
    ax_2.view_init(azim = 75, elev = 20)
    plot_2 = ax_2.scatter(
        datasets[l]['reconstructed1'][:, 0],
        datasets[l]['reconstructed1'][:, 1],
        datasets[l]['reconstructed1'][:, 2],
        c = colors[1],
        alpha = 0.3,
        )
    
    #2-dimensional compression
    ax_3 = fig.add_subplot(gs_2[-1+2], projection = '3d')
    ax_3.set_box_aspect([1, 1, 1])
    ax_3.set_xlim(*xlim)
    ax_3.set_ylim(*ylim)
    ax_3.set_zlim(*zlim)
    ax_3.set_xticklabels([])
    ax_3.set_yticklabels([])
    ax_3.set_zticklabels([])
    ax_3.view_init(azim = 75, elev = 20)
    plot_3 = ax_3.scatter(
        datasets[l]['reconstructed2'][:, 0],
        datasets[l]['reconstructed2'][:, 1],
        datasets[l]['reconstructed2'][:, 2],
        c = colors[2],
        alpha = 0.3,
        )

    fig.savefig(f'figures/{l}_plateau.png', dpi = 300)

del fig, gs, gs_2, training_losses, data, autoencoder, begin, history, end, lapse, reconstructed, xlim, ylim, zlim, ax_2, plot_2, ax_3, plot_3

del bottlenecks, epochs, batch_size, colors


for l in datasets.keys():
    print(f'Dataset: {l}\n')
    
    #estimations
    print('Estimated: {estimated}'.format(estimated = datasets[l]['estimated']))

    #lapses
    print('Lapses:')
    for ll in datasets[l]['lapses'].keys():
        print('  {part}: {duration:.2f} s'.format(
            part = ll,
            duration = datasets[l]['lapses'][ll],
            ))
    print('\n')
