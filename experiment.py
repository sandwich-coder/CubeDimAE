## CubeDimAE

import itertools
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (6, 6)
from sklearn.datasets import make_s_curve, make_swiss_roll
import tensorflow as tf
from tensorflow.keras import layers
import time

from cube_dim import CubeDim

#preparatory functions
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
def limit(data):
    ranges = data.max(axis = 0) - data.min(axis = 0)
    width = ranges.max().tolist()
    width = width * 1.2    #margins
    box = np.array([[0] * 3,
                    [width] * 3], dtype = 'float64')
    
    middle_box = (box.max(axis = 0) + box.min(axis = 0)) / np.float64(2)
    middle_data = (data.max(axis = 0) + data.min(axis = 0)) / np.float64(2)
    move = middle_data - middle_box
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


# - - construction - -

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
            raise AttributeError('The dataset is not of \'tensorflow.float32\'.')
        
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
        


# - - data - -

#s curve <2-dimensional)>
data1 = make_s_curve(n_samples = 3200, random_state = 1)[0]

#swiss roll <2-dimensional>
data2 = make_swiss_roll(n_samples = 3200, random_state = 1)[0]

#mobius strip (from kaggle) <2-dimensional>
theta = np.linspace(0, 2 * np.pi, num = 200, endpoint = False, dtype = 'float64')
w = np.linspace(-0.25, 0.25, 16)
w, theta = np.meshgrid(w, theta)
phi = -0.5 * theta    # multiple of 0.5
r = 1 + w * np.cos(phi)
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))
data3 = np.stack([x, y, z], axis = 1)
del theta, w, phi, r, x, y, z

#hollow sphere <2-dimensional>
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
data4 = np.stack([x1, x2, x3], axis = 1)
del radius, azimuth, elevation, x1, x2, x3

#solid sphere <3-dimensional>
displacement = np.random.uniform(low = 0, high = 1, size = [3200]) ** np.float64(1 / 3)
azimuth = np.random.uniform(low = -np.pi, high = np.pi, size = [3200])
elevation = np.random.uniform(low = -1, high = 1, size = [3200])
elevation = np.arcsin(elevation, dtype = 'float64')
x1 = displacement * np.cos(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x2 = displacement * np.sin(azimuth, dtype = 'float64') * np.cos(elevation, dtype = 'float64')
x3 = displacement * np.sin(elevation, dtype = 'float64')
data5 = np.stack([x1, x2, x3], axis = 1)

for l in [data1, data2, data3, data4, data5]:
    _xlim, _ylim, _zlim = limit(l)
    _fig = plt.figure(layout = 'constrained')
    _axes = _fig.add_subplot(projection = '3d')
    _axes.set_box_aspect([1, 1, 1])
    _axes.set_xlim(*_xlim)
    _axes.set_ylim(*_ylim)
    _axes.set_zlim(*_zlim)
    _axes.view_init(azim = 75, elev = 20)
    _plot = _axes.scatter(l[:, 0], l[:, 1], l[:, 2], alpha = 0.3)



' ========== estimation ========== '


# - - estimation - -

_begin = time.time()
dimension1, parse1 = estimator.estimate(data1, return_parse = True)
_end = time.time()
duration1 = _end - _begin

_begin = time.time()
dimension2, parse2 = estimator.estimate(data2, return_parse = True)
_end = time.time()
duration2 = _end - _begin

_begin = time.time()
dimension3, parse3 = estimator.estimate(data3, return_parse = True)
_end = time.time()
duration3 = _end - _begin

_begin = time.time()
dimension4, parse4 = estimator.estimate(data4, return_parse = True)
_end = time.time()
duration4 = _end - _begin

_begin = time.time()
dimension5, parse5 = estimator.estimate(data5, return_parse = True)
_end = time.time()
duration5 = _end - _begin

#parsing
parses = [parse1, parse2, parse3, parse4, parse5]
_fig = plt.figure(layout = 'constrained')
_gs = _fig.add_gridspec(nrows = len(parses), ncols = 1)
for l in range(len(parses)):
    _lengths = parses[l][:, 0].copy()
    _connections = parses[l][:, 1].copy()
    _dimensions_ = np.log(np.float64(1) + _connections) / np.log(np.float64(3))
    _axes = _fig.add_subplot(_gs[l])
    _axes.set_box_aspect(0.2)
    _plot = _axes.plot(_lengths, _dimensions_, color = 'green')



' ========== training and comparison ========== '


# - - training - -

#to Tensors
data1 = tf.constant(data1, dtype = 'float32')
data2 = tf.constant(data2, dtype = 'float32')
data3 = tf.constant(data3, dtype = 'float32')
data4 = tf.constant(data4, dtype = 'float32')
data5 = tf.constant(data5, dtype = 'float32')

#autoencoders
latent = [1, 2]
optional = [3, 4, 5]
autoencoders = {}
for l in latent + optional:
    autoencoders[l] = Autoencoder(l)
    autoencoders[l].compile(optimizer = 'adam', loss = 'mse')

#learning
histories1, elapsed1, reconstructions1 = {}, {}, {}
histories2, elapsed2, reconstructions2 = {}, {}, {}
histories3, elapsed3, reconstructions3 = {}, {}, {}
histories4, elapsed4, reconstructions4 = {}, {}, {}
histories5, elapsed5, reconstructions5 = {}, {}, {}
for l in latent + optional:
    
    _begin = time.time()
    histories1[l] = autoencoders[l].fit(data1, data1, batch_size = 32, epochs = 50, shuffle = True)
    _end = time.time()
    elapsed1[l] = _end - _begin
    reconstructions1[l] = autoencoders[l].call(data1)
    
    _begin = time.time()
    histories2[l] = autoencoders[l].fit(data2, data2, batch_size = 32, epochs = 50, shuffle = True)
    _end = time.time()
    elapsed2[l] = _end - _begin
    reconstructions2[l] = autoencoders[l].call(data2)
    
    _begin = time.time()
    histories3[l] = autoencoders[l].fit(data3, data3, batch_size = 32, epochs = 50, shuffle = True)
    _end = time.time()
    elapsed3[l] = _end - _begin
    reconstructions3[l] = autoencoders[l].call(data3)
    
    _begin = time.time()
    histories4[l] = autoencoders[l].fit(data4, data4, batch_size = 32, epochs = 50, shuffle = True)
    _end = time.time()
    elapsed4[l] = _end - _begin
    reconstructions4[l] = autoencoders[l].call(data4)
    
    _begin = time.time()
    histories5[l] = autoencoders[l].fit(data5, data5, batch_size = 32, epochs = 50, shuffle = True)
    _end = time.time()
    elapsed5[l] = _end - _begin
    reconstructions5[l] = autoencoders[l].call(data5)
    


# - - plot - -

fig1 = plt.figure(layout = 'constrained')
_gs = fig1.add_gridspec(nrows = 2, ncols = 1)
_gs_1 = _gs[1].subgridspec(nrows = 1, ncols = 2)

axes1_1 = fig1.add_subplot(_gs[0])
axes1_1.set_box_aspect(0.5)
axes1_1.set_title('Training Loss')
axes1_1.set_xlabel('epoch')
axes1_1.set_ylabel('loss')
plot1_1_1 = axes1_1.plot(histories1[1].history['loss'], color = 'red', label = '1-dimensional')
plot1_1_2 = axes1_1.plot(histories1[2].history['loss'], color = 'green', label = '2-dimensional')
plot1_1_3 = axes1_1.plot(histories1[3].history['loss'], color = 'burlywood', label = '3-dimensional')
plot1_1_4 = axes1_1.plot(histories1[4].history['loss'], color = 'cyan', label = '4-dimensional')
plot1_1_5 = axes1_1.plot(histories1[5].history['loss'], color = 'pink', label = '5-dimensional')
axes1_1.legend()

_xlim, _ylim, _zlim = limit(data1.numpy())
axes1_2 = fig1.add_subplot(_gs_1[0], projection = '3d')
axes1_2.set_box_aspect([1, 1, 1])
axes1_2.set_xlim(*_xlim)
axes1_2.set_ylim(*_ylim)
axes1_2.set_zlim(*_zlim)
axes1_2.set_xticklabels([])
axes1_2.set_yticklabels([])
axes1_2.set_zticklabels([])
axes1_2.view_init(azim = 75, elev = 20)
plot1_2 = axes1_2.scatter(reconstructions1[1][:, 0], reconstructions1[1][:, 1], reconstructions1[1][:, 2], c = 'red', alpha = 0.3)
axes1_3 = fig1.add_subplot(_gs_1[1], projection = '3d')
axes1_3.set_box_aspect([1, 1, 1])
axes1_3.set_xlim(*_xlim)
axes1_3.set_ylim(*_ylim)
axes1_3.set_zlim(*_zlim)
axes1_3.set_xticklabels([])
axes1_3.set_yticklabels([])
axes1_3.set_zticklabels([])
axes1_3.view_init(azim = 75, elev = 20)
plot1_3 = axes1_3.scatter(reconstructions1[2][:, 0], reconstructions1[2][:, 1], reconstructions1[1][:, 2], c = 'green', alpha = 0.3)

fig2 = plt.figure(layout = 'constrained')
_gs = fig2.add_gridspec(nrows = 2, ncols = 1)
_gs_1 = _gs[1].subgridspec(nrows = 1, ncols = 2)

axes2_1 = fig2.add_subplot(_gs[0])
axes2_1.set_box_aspect(0.5)
axes2_1.set_title('Training Loss')
plot2_1_1 = axes2_1.plot(histories2[1].history['loss'], color = 'red', label = '1-dimensional compression')
plot2_1_2 = axes2_1.plot(histories2[2].history['loss'], color = 'green', label = '2-dimensional compression')
plot2_1_3 = axes2_1.plot(histories2[3].history['loss'], color = 'burlywood', label = '3-dimensional')
plot2_1_4 = axes2_1.plot(histories2[4].history['loss'], color = 'cyan', label = '4-dimensional')
plot2_1_5 = axes2_1.plot(histories2[5].history['loss'], color = 'pink', label = '5-dimensional')
axes2_1.legend()

_xlim, _ylim, _zlim = limit(data2.numpy())
axes2_2 = fig2.add_subplot(_gs_1[0], projection = '3d')
axes2_2.set_box_aspect([1, 1, 1])
axes2_2.set_xlim(*_xlim)
axes2_2.set_ylim(*_ylim)
axes2_2.set_zlim(*_zlim)
axes2_2.set_xticklabels([])
axes2_2.set_yticklabels([])
axes2_2.set_zticklabels([])
axes2_2.view_init(azim = 75, elev = 20)
plot2_2 = axes2_2.scatter(reconstructions2[1][:, 0], reconstructions2[1][:, 1], reconstructions2[1][:, 2], c = 'red', alpha = 0.3)
axes2_3 = fig2.add_subplot(_gs_1[1], projection = '3d')
axes2_3.set_box_aspect([1, 1, 1])
axes2_3.set_xlim(*_xlim)
axes2_3.set_ylim(*_ylim)
axes2_3.set_zlim(*_zlim)
axes2_3.set_xticklabels([])
axes2_3.set_yticklabels([])
axes2_3.set_zticklabels([])
axes2_3.view_init(azim = 75, elev = 20)
plot2_3 = axes2_3.scatter(reconstructions2[2][:, 0], reconstructions2[2][:, 1], reconstructions2[2][:, 2], c = 'green', alpha = 0.3)

fig3 = plt.figure(layout = 'constrained')
_gs = fig3.add_gridspec(nrows = 2, ncols = 1)
_gs_1 = _gs[1].subgridspec(nrows = 1, ncols = 2)

axes3_1 = fig3.add_subplot(_gs[0])
axes3_1.set_box_aspect(0.5)
axes3_1.set_title('Training Loss')
plot3_1_1 = axes3_1.plot(histories3[1].history['loss'], color = 'red', label = '1-dimensional compression')
plot3_1_2 = axes3_1.plot(histories3[2].history['loss'], color = 'green', label = '2-dimensional compression')
plot3_1_3 = axes3_1.plot(histories3[3].history['loss'], color = 'burlywood', label = '3-dimensional')
plot3_1_4 = axes3_1.plot(histories3[4].history['loss'], color = 'cyan', label = '4-dimensional')
plot3_1_5 = axes3_1.plot(histories3[5].history['loss'], color = 'pink', label = '5-dimensional')
axes3_1.legend()

_xlim, _ylim, _zlim = limit(data3.numpy())
axes3_2 = fig3.add_subplot(_gs_1[0], projection = '3d')
axes3_2.set_box_aspect([1, 1, 1])
axes3_2.set_xlim(*_xlim)
axes3_2.set_ylim(*_ylim)
axes3_2.set_zlim(*_zlim)
axes3_2.set_xticklabels([])
axes3_2.set_yticklabels([])
axes3_2.set_zticklabels([])
axes3_2.view_init(azim = 75, elev = 20)
plot3_2 = axes3_2.scatter(reconstructions3[1][:, 0], reconstructions3[1][:, 1], reconstructions3[1][:, 2], c = 'red', alpha = 0.3)
axes3_3 = fig3.add_subplot(_gs_1[1], projection = '3d')
axes3_3.set_box_aspect([1, 1, 1])
axes3_3.set_xlim(*_xlim)
axes3_3.set_ylim(*_ylim)
axes3_3.set_zlim(*_zlim)
axes3_3.set_xticklabels([])
axes3_3.set_yticklabels([])
axes3_3.set_zticklabels([])
axes3_3.view_init(azim = 75, elev = 20)
plot3_3 = axes3_3.scatter(reconstructions3[2][:, 0], reconstructions3[2][:, 1], reconstructions3[2][:, 2], c = 'green', alpha = 0.3)

fig4 = plt.figure(layout = 'constrained')
_gs = fig4.add_gridspec(nrows = 2, ncols = 1)
_gs_1 = _gs[1].subgridspec(nrows = 1, ncols = 2)

axes4_1 = fig4.add_subplot(_gs[0])
axes4_1.set_box_aspect(0.5)
axes4_1.set_title('Training Loss')
plot4_1_1 = axes4_1.plot(histories4[1].history['loss'], color = 'red', label = '1-dimensional compression')
plot4_1_2 = axes4_1.plot(histories4[2].history['loss'], color = 'green', label = '2-dimensional compression')
plot4_1_3 = axes4_1.plot(histories4[3].history['loss'], color = 'burlywood', label = '3-dimensional')
plot4_1_4 = axes4_1.plot(histories4[4].history['loss'], color = 'cyan', label = '4-dimensional')
plot4_1_5 = axes4_1.plot(histories4[5].history['loss'], color = 'pink', label = '5-dimensional')
axes4_1.legend()

_xlim, _ylim, _zlim = limit(data4.numpy())
axes4_2 = fig4.add_subplot(_gs_1[0], projection = '3d')
axes4_2.set_box_aspect([1, 1, 1])
axes4_2.set_xlim(*_xlim)
axes4_2.set_ylim(*_ylim)
axes4_2.set_zlim(*_zlim)
axes4_2.set_xticklabels([])
axes4_2.set_yticklabels([])
axes4_2.set_zticklabels([])
axes4_2.view_init(azim = 75, elev = 20)
plot4_2 = axes4_2.scatter(reconstructions4[1][:, 0], reconstructions4[1][:, 1], reconstructions4[1][:, 2], c = 'red', alpha = 0.3)
axes4_3 = fig4.add_subplot(_gs_1[1], projection = '3d')
axes4_3.set_box_aspect([1, 1, 1])
axes4_3.set_xlim(*_xlim)
axes4_3.set_ylim(*_ylim)
axes4_3.set_zlim(*_zlim)
axes4_3.set_xticklabels([])
axes4_3.set_yticklabels([])
axes4_3.set_zticklabels([])
axes4_3.view_init(azim = 75, elev = 20)
plot4_3 = axes4_3.scatter(reconstructions4[2][:, 0], reconstructions4[2][:, 1], reconstructions4[2][:, 2], c = 'green', alpha = 0.3)

fig5 = plt.figure(layout = 'constrained')
_gs = fig5.add_gridspec(nrows = 2, ncols = 1)
_gs_1 = _gs[1].subgridspec(nrows = 1, ncols = 2)

axes5_1 = fig5.add_subplot(_gs[0])
axes5_1.set_box_aspect(0.5)
axes5_1.set_title('Training Loss')
plot5_1_1 = axes5_1.plot(histories5[1].history['loss'], color = 'red', label = '1-dimensional compression')
plot5_1_2 = axes5_1.plot(histories5[2].history['loss'], color = 'green', label = '2-dimensional compression')
plot5_1_3 = axes5_1.plot(histories5[3].history['loss'], color = 'burlywood', label = '3-dimensional')
plot5_1_4 = axes5_1.plot(histories5[4].history['loss'], color = 'cyan', label = '4-dimensional')
plot5_1_5 = axes5_1.plot(histories5[5].history['loss'], color = 'pink', label = '5-dimensional')
axes5_1.legend()

_xlim, _ylim, _zlim = limit(data5.numpy())
axes5_2 = fig5.add_subplot(_gs_1[0], projection = '3d')
axes5_2.set_box_aspect([1, 1, 1])
axes5_2.set_xlim(*_xlim)
axes5_2.set_ylim(*_ylim)
axes5_2.set_zlim(*_zlim)
axes5_2.set_xticklabels([])
axes5_2.set_yticklabels([])
axes5_2.set_zticklabels([])
axes5_2.view_init(azim = 75, elev = 20)
plot5_2 = axes5_2.scatter(reconstructions5[1][:, 0], reconstructions5[1][:, 1], reconstructions5[1][:, 2], c = 'red', alpha = 0.3)
axes5_3 = fig5.add_subplot(_gs_1[1], projection = '3d')
axes5_3.set_box_aspect([1, 1, 1])
axes5_3.set_xlim(*_xlim)
axes5_3.set_ylim(*_ylim)
axes5_3.set_zlim(*_zlim)
axes5_3.set_xticklabels([])
axes5_3.set_yticklabels([])
axes5_3.set_zticklabels([])
axes5_3.view_init(azim = 75, elev = 20)
plot5_3 = axes5_3.scatter(reconstructions5[2][:, 0], reconstructions5[2][:, 1], reconstructions5[2][:, 2], c = 'green', alpha = 0.3)