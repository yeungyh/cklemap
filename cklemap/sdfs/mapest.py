import numpy as np
import scipy.sparse as sps

def compute_Lreg(geom):
    Nc        = geom.cells.num
    Nfint     = geom.faces.num_interior)
    neighbors = geom.faces.neighbors[:, Nfint:]
    rows      = np.concatenate((np.arange(Nfint), np.arange(Nfint)))
    cols      = np.concatenate((neighbors[0], neighbors[1]))
    vals      = np.concatenate((np.full(Nfint, -1.0), np.full(Nfint, 1.0)))
    return sps.coo_matrix((vals, (rows, cols)), shape=(Nfint, Nc))

class LossVec(object):

    def __init__(self, iuobs, uobs, iYobs, Yobs, gamma, L):
        self.iuobs = iuobs
        self.uobs  = uobs
        self.iYobs = iYobs
        self.Yobs  = Yobs
        self.gamma = gamma
        self.L     = L

    def val(self, u, Y):
        Ly  = self.L.dot(Y)
        vec = np.concatenate(((self.uobs - u[self.iuobs]).reshape(-1), (self.Yobs - Y[self.iYobs]).reshape(-1), np.sqrt(self.gamma) * Ly.reshape(-1)))
        return vec

    def grad_u(self, u, Y):
        Nus  = self.iuobs.size
        Nu   = u.size
        cols = self.iuobs
        rows = np.arange(Nus)
        vals = np.full(Nus, 1.0)
        Hu   = sps.coo_matrix((vals, (rows, cols)), shape=(Nus, Nu))
        return -Hu

    def grad_Y(self, u, Y):
        NYs  = self.iYobs.size
        NY   = Y.size
        cols = self.iYobs
        rows = np.arange(NYs)
        vals = np.full(NYs, 1.0)
        Hy   = sps.coo_matrix((vals, (rows, cols)), shape=(NYs, NY))
        return sps.vstack([-Hy, np.sqrt(self.gamma) * self.L])
