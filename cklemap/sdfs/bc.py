import numpy as np

class BC(object):

    def __init__(self, geom):
        self.Ni = geom.faces.num_interior
        self.kind = np.empty(geom.faces.num - self.Ni, dtype="str")
        self.val = np.zeros(geom.faces.num - self.Ni)

    def side_gf_idx(self, geom, side):

        if side == "left":
            idx = geom.faces.centroids[0, self.Ni:] == 0.0
        elif side == "right":
            idx = geom.faces.centroids[0, self.Ni:] == geom.L[0]
        elif side == "bottom":
            idx = geom.faces.centroids[1, self.Ni:] == 0.0
        elif side == "top":
            idx = geom.faces.centroids[1, self.Ni:] == geom.L[1]

        return idx

    def dirichlet(self, geom, side, val):

        idx = self.side_gf_idx(geom, side)
        self.kind[idx] = "D"
        self.val[idx]  = val
