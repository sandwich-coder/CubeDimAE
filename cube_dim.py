import itertools
import numpy as np
from tqdm import tqdm


class CubeDim:
    def __init__(self):
        pass
    def __repr__(self):
        return 'cubeDim'
    def estimate(self, data, trim = None, split = None, return_parse = False):
        if type(data) != np.ndarray:
            raise TypeError('Dataset is not a \'numpy.ndarray\'.')
        if data.dtype != np.float64:
            raise AttributeError('Dataset is not of \'numpy.float64\'.')
        if trim != None:
            if not isinstance(trim, float):
                raise TypeError('\'trim\' should be a float.')
            if not 0 < trim < 1:
                raise ValueError('\'trim\' must be between 0 and 1.')
        if split != None:
            if not isinstance(split, int):
                raise TypeError('\'split\' should be an integer.')
            if split < 1:
                raise ValueError('\'split\' must be greater than 0.')
        if return_parse != False:
            if not isinstance(return_parse, bool):
                raise TypeError('\'return_parse\' should be boolean.')
        
        #trimmed
        if trim != None:
            trim = trim ** (1 / data.shape[1])
            low = np.quantile(data, (1 - trim) / 2, axis = 0)
            high = np.quantile(data, (1 + trim) / 2, axis = 0)
            leftcut = data >= low.reshape([-1, low.shape[0]])    # Equality takes into account the rounding errors.
            rightcut = data <= high.reshape([-1, high.shape[0]])
            indices = leftcut & rightcut
            
            indices = np.logical_and.reduce(indices, axis = 1, dtype = 'bool')
            data = data[indices, :]
        
        #centered
        data = data - data.mean(axis = 0, dtype = 'float64', keepdims = True)
        
        #length
        mins = data.min(axis = 0)
        maxs = data.max(axis = 0)
        scale = np.max(maxs - mins, axis = 0)
        lengths = np.linspace(scale * np.float64(0.01), scale * np.float64(0.1), num = 10, dtype = 'float64')
        
        connections = []
        for lll in tqdm(lengths, colour = 'magenta', ncols = 70):
            nodes = np.round(data / lll)
            nodes = nodes.astype('int8', copy = False)
            nodes, counts = np.unique(nodes, axis = 0, return_counts = True)
            
            indices = itertools.product(range(nodes.shape[0]), range(nodes.shape[0]))
            indices = np.transpose(np.array(list(indices), dtype = 'int64'))
            
            
            if split != None:
                is_adjacent = []
                for lllll in itertools.batched(range(nodes.shape[1]), n = split):
                    _nodes = nodes[:, lllll].copy()
                    _nodes_product = np.empty([_nodes.shape[0] ** 2, 2, _nodes.shape[1]], dtype = 'int8')
                    _nodes_product[:, 0, :] = _nodes[indices[0]].copy()
                    _nodes_product[:, 1, :] = _nodes[indices[1]].copy()
                    _nodes_product = _nodes_product.reshape([_nodes.shape[0], _nodes.shape[0], 2, _nodes.shape[1]])
                    is_adjacent.append(np.max(np.absolute(_nodes_product[:, :, 1, :] - _nodes_product[:, :, 0, :]), axis = 2))
                is_adjacent = np.stack(is_adjacent, axis = 2)
                is_adjacent = is_adjacent.max(axis = 2) == np.int8(1)
            
            
            else:
                nodes_product = np.empty([nodes.shape[0] ** 2, 2, nodes.shape[1]], dtype = 'int8')
                nodes_product[:, 0, :] = nodes[indices[0]].copy()
                nodes_product[:, 1, :] = nodes[indices[1]].copy()
                nodes_product = nodes_product.reshape([nodes.shape[0], nodes.shape[0], 2, nodes.shape[1]])
                is_adjacent = np.max(np.absolute(nodes_product[:, :, 1, :] - nodes_product[:, :, 0, :]), axis = 2) == np.int8(1)    #integers
            
            neighbors = is_adjacent.astype('int64', copy = True)
            
            adjacencies = neighbors.sum(axis = 1, dtype = 'float64')
            connection = np.average(adjacencies, weights = counts)    # 'Connection' is a weighted average of adjacency.
            connections.append(connection)
            
        connections = np.stack(connections)
        
        dimension = connections.max(axis = 0)
        dimension = np.log(dimension + np.float64(1)) / np.log(np.float64(3))
        dimension = dimension.round().astype('int64', copy = False)
        dimension = dimension.tolist()
        
        if return_parse:
            parse = np.stack([lengths, connections], axis = 1)
            return dimension, parse
        else:
            return dimension
