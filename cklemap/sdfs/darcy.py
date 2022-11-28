import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse as sps
import scipy.io as spio
from cklemap.sdfs.amps import AMPS, AMPS_full
from time import perf_counter


class DarcyExp(object):

    def __init__(self, tpfa, iuobs, use_amps, ssv=None):
        self.tpfa = tpfa
        self.Nc = self.tpfa.geom.cells.num
        self.Nc_range = np.arange(self.Nc)
        self.ssv = range(self.tpfa.geom.cells.num) if ssv is None else ssv
        self.cells_neighbors = self.tpfa.cell_neighbors
        self.keep = np.concatenate(
            (self.Nc_range, np.flatnonzero(self.cells_neighbors >= 0) + self.Nc))
        self.cols = np.concatenate(
            (self.Nc_range, np.tile(self.Nc_range, 4)))[self.keep]
        self.rows = np.concatenate(
            (self.Nc_range, self.cells_neighbors.ravel()))[self.keep]
        self.rows_old = np.repeat(self.Nc_range, 5)
        neumann_bc = (self.tpfa.bc.kind == 'N')
        Nq = np.count_nonzero(neumann_bc)
        self.dLdq = sps.csc_matrix(
            (-np.ones(Nq), (np.arange(Nq), self.tpfa.geom.cells.to_hf[2*self.tpfa.Ni:][neumann_bc])), shape=(Nq, self.Nc))
        #self.dLdp = sps.csc_matrix((np.ones(self.Nc + self.keep.size), (self.rows, self.cols)), shape=(self.Nc, self.Nc))
        self.use_amps = use_amps
        self.setup_amps(iuobs)
        if self.use_amps:
            self.rows = self.amps.Pt[self.rows]

    def randomize_bc(self, kind, scale):
        self.tpfa.bc.randomize(kind, scale)
        self.tpfa.update_rhs(kind)
        return self

    def increment_bc(self, kind, value):
        self.tpfa.bc.increment(kind, value)
        self.tpfa.update_rhs(kind)
        return self

    def setup_amps(self, iuobs):
        self.iuobs = iuobs
        if self.use_amps:
            self.amps = AMPS(sps.csc_matrix((np.ones(2 * self.tpfa.Ni + self.Nc), (self.tpfa.rows, self.tpfa.cols)), shape=(self.Nc, self.Nc)),
                             self.tpfa.rhs_dirichlet + self.tpfa.rhs_neumann, iuobs)
            # self.amps = AMPS(sps.csc_matrix((np.ones(2 * self.tpfa.Ni + self.Nc), (self.tpfa.rows, self.tpfa.cols)), shape=(self.Nc, self.Nc)),
            #                  self.tpfa.rhs_dirichlet + self.tpfa.rhs_neumann, iuobs,
            #                  sps.coo_matrix((np.ones(self.keep.size), (self.rows, self.cols)), shape=(self.Nc, self.Nc)))
        else:
            self.amps = AMPS_full(sps.csc_matrix((np.ones(
                2 * self.tpfa.Ni + self.Nc), (self.tpfa.rows, self.tpfa.cols)), shape=(self.Nc, self.Nc)), self.iuobs.size)

    def construct_pde(self, Y, q=None):
        self.A, b = self.tpfa.ops(np.exp(Y), q)
        return b

    def solve(self, Y, q=None):
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K, q)
        if self.amps is not None:
            self.amps.update(self.A)
            return self.amps.solve_pde(b)
        else:
            return spl.spsolve(self.A, b)

    def adj_solve(self, B):
        if self.amps is not None:
            return self.amps.solve_jac(B)
        else:
            return spl.spsolve(self.A, B)

    def partial_solve(self, Y):
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K)
        if self.amps is not None:
            #time_start = perf_counter()
            self.amps.update(self.A)
            #print(f'update time elasped: {perf_counter() - time_start}')
            #time_start = perf_counter()
            sol = self.amps.solve_pde(b)
            #print(f'amps time elasped: {perf_counter() - time_start}')
            #time_start = perf_counter()
            #sol_cmp = spl.spsolve(self.A, b)
            #print(f'spsolve time elasped: {perf_counter() - time_start}')
            #print(np.allclose(sol, sol_cmp))
            return sol
        else:
            return spl.spsolve(self.A, b)

    def residual(self, u, Y):
        self.K = np.exp(Y)
        self.A, b = self.tpfa.ops(self.K)
        return self.A @ u - b

    def residual_sens_Y_old(self, u, Y):
        # call residual(self, u, Y) before residual_sens_Y(self, u, Y)
        diag, offdiags, cols, bsens = self.tpfa.sens_old()
        vals = np.hstack(((diag * u - (offdiags * u[cols]).sum(axis=1) - bsens)[:, None],
                          (u[cols] - u[:, None]) * offdiags)) * np.exp(Y)[:, None]
        cols = np.hstack((self.Nc_range[:, None], cols))
        keep = np.flatnonzero(cols >= 0)
        return sps.csc_matrix((vals.ravel()[keep], (self.rows_old[keep], cols.ravel()[keep])), shape=(self.Nc, self.Nc))

    def residual_sens_Y(self, u, Y):
        # call residual(self, u, Y) before residual_sens_Y(self, u, Y)
        offdiags = (u[self.cells_neighbors] - u[None, :]) * self.tpfa.sens()
        vals = np.vstack(((self.tpfa.alpha_dirichlet * u - self.tpfa.rhs_dirichlet -
                           offdiags.sum(axis=0))[None, :], offdiags)) * self.K[None, :]
        return sps.csc_matrix((vals.ravel()[self.keep], (self.rows, self.cols)), shape=(self.Nc, self.Nc))

    def residual_sens_u(self, u, Y):
        # call residual(self, u, Y) before residual_sens_u(self, u, Y)
        return self.A

    def residual_sens_p_old(self, u, p):
        # call residual(self, u, Y) before residual_sens_p(self, u, p)
        return sps.vstack([self.residual_sens_Y_old(u, p[:self.tpfa.geom.cells.num]), self.dLdq])

    def residual_sens_p(self, u, p):
        # call residual(self, u, Y) before residual_sens_p(self, u, p)
        return sps.vstack([self.residual_sens_Y(u, p[:self.tpfa.geom.cells.num]), self.dLdq])

    def u_sens_p(self, dLdp):
        if self.amps is not None:
            #            time_start = perf_counter()
            # self.amps.update(self.A)
            #            print(f'update time elasped: {perf_counter() - time_start}')
            #            time_start = perf_counter()
            return self.amps.solve_jac().T @ dLdp
#            print(f'amps time elasped: {perf_counter() - time_start}')
#            return dudp
        else:
            return spl.spsolve(self.A, sps.csc_matrix((np.ones(self.iuobs.size), (self.iuobs, np.arange(self.iuobs.size))), shape=(self.Nu, self.iuobs.size))).T @ dLdp
            # return spl.spsolve(self.A, dLdp)[self.iuobs, :]


class DarcyExpTimeDependent(DarcyExp):

    def __init__(self, tpfa, ss, dt, ssv=None):
        super().__init__(tpfa, ssv)
        self.ss = ss
        self.dt = dt
        self.c = self.ss / self.dt
        self.C = self.c * sps.eye(self.Nc)
        self.prev_u = np.zeros(self.Nc)
        self.c_prev_u = self.c * self.prev_u

    def update_u(self, prev_u):
        self.prev_u = prev_u
        self.c_prev_u = self.c * self.prev_u

    def solve(self, Y, q=None):
        self.A, b = self.tpfa.ops(np.exp(Y), q)
        return spl.spsolve(self.A - self.C, b - self.c_prev_u)

    def residual(self, u, Y):
        self.A, b = self.tpfa.ops(np.exp(Y))
        return self.A @ u - b - self.c * (u - self.prev_u)

    def residual_sens_u(self, u, Y):
        return self.A - self.C
