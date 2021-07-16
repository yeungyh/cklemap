import numpy as np
from collections import namedtuple
import time

class Geom(object):

    def __init__(self, L, N):
        self.L = L
        self.N = N
        # Cartesian coordinates
        self.d = self.L / (1.0 * self.N)
        self.x = np.linspace(self.d[0] / 2, self.L[0] - self.d[0] / 2, self.N[0])
        self.y = np.linspace(self.d[1] / 2, self.L[1] - self.d[1] / 2, self.N[1])
        # Substructures
        self.faces = namedtuple('faces', ['num', 'centroids', 'to_hf', 'areas', 'normals', 'neighbors', 'num_interior', 'int_ext'])
        self.cells = namedtuple('cells', ['num', 'centroids', 'to_hf', 'volumes', 'half_faces'])

    def calculate(self):
        sx = self.N[0] + 1
        sy = self.N[1] + 1
        nfx = sx * self.N[1]
        nfy = sy * self.N[0]
        self.faces.num = nfx + nfy
        self.cells.num = np.prod(self.N)
        
        ## Neighbors
        # -1 indicates that there's no neighbor
        C = np.pad(np.arange(self.cells.num).reshape((self.N[1], -1)), 1, 'constant', constant_values=-1)
        self.faces.neighbors = np.block([[C[1:sy, :sx].ravel(), C[:sy, 1:sx].ravel()],
                                         [C[1:sy, 1: ].ravel(), C[1:,  1:sx].ravel()]])
        
        ## Interior face identification and sort
        is_interior = np.logical_and(*(self.faces.neighbors >= 0))
        self.faces.num_interior = np.count_nonzero(is_interior)
        self.faces.int_ext = np.argsort(~is_interior)
        self.faces.neighbors = self.faces.neighbors[:, self.faces.int_ext]
        
        ## Global faces to half faces
        Ni_range = np.arange(self.faces.num_interior)
        self.faces.to_hf = np.concatenate((Ni_range, Ni_range, np.arange(self.faces.num_interior, self.faces.num)))
        
        ## Face Centroids
        self.faces.centroids = np.hstack((np.asarray(np.meshgrid(np.linspace(0.0, self.d[0] * self.N[0], sx), self.y)).reshape(2, -1),
                                          np.asarray(np.meshgrid(self.x, np.linspace(0.0, self.d[1] * self.N[1], sy))).reshape(2, -1)))[:, self.faces.int_ext]
        
        ## Areas
        self.faces.areas = np.concatenate((np.repeat(self.d[1], nfx), np.repeat(self.d[0], nfy)))
        
        ## Normals
        self.faces.normals = np.block([[np.full(nfx, self.d[1]), np.full(nfy, 0)],
                                       [np.full(nfx, 0),         np.full(nfy, self.d[0])]])[:, self.faces.int_ext]
        self.faces.normals[:, self.faces.num_interior:] *= np.array([1, -1]).dot(self.faces.neighbors[:, self.faces.num_interior:] >= 0)

        ## Cell Centroids
        self.cells.centroids = np.asarray(np.meshgrid(self.x, self.y)).reshape(2, -1)
        
        ## Cells to half faces
        self.cells.to_hf = np.concatenate((self.faces.neighbors[:, :self.faces.num_interior].ravel(),
                                           np.max(self.faces.neighbors[:, self.faces.num_interior:], axis=0)))

        ## Volumes
        self.cells.volumes = np.repeat(np.prod(self.d), self.cells.num)

        ## Cell faces
        self.cells.half_faces = np.argsort(self.cells.to_hf).reshape(-1, 4)