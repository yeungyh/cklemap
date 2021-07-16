from os import pathsep
from re import X
from numba import njit, prange
from numba.np.ufunc import parallel
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import sksparse.cholmod as cm
from time import perf_counter

@njit
def partial_spsolve_pde_right(L, iL, jL, b, iter):
    for j in iter:
        b[j] /= L[jL[j]]
        b[iL[jL[j]+1:jL[j+1]]] -= L[jL[j]+1:jL[j+1]] * b[j]

@njit
def partial_spsolve_pde_left(L, iL, jL, b, iter):
    for j in iter:
        b[j] -= np.sum(L[jL[j]+1:jL[j+1]] * b[iL[jL[j]+1:jL[j+1]]])
        b[j] /= L[jL[j]]

@njit(parallel=True)
def partial_spsolve_jac_right(L, iL, jL, B, iter, iter_lens):
    for i in prange(B.shape[1]):
        for j in iter[i, :iter_lens[i]]:
            B[j, i] /= L[jL[j]]
            B[iL[jL[j]+1:jL[j+1]], i] -= L[jL[j]+1:jL[j+1]] * B[j, i]

@njit
def partial_spsolve_jac_left(L, iL, jL, B, iter):
    for j in iter:
        B[j, :] -= L[jL[j]+1:jL[j+1]] @ B[iL[jL[j]+1:jL[j+1]], :]
        B[j, :] /= L[jL[j]]
        

class AMPS(object):

    def __init__(self, A, nz_left, nz_right_pde, nz_right_jac):
        self.factor = cm.analyze(A, mode="auto", ordering_method="nesdis", use_long=False)
        self.n = A.shape[0]
        self.P = self.factor.P()
        self.L = None
        self.inz_left = np.ravel(nz_left.sum(axis=1)).astype(bool)
        self.closure_left = np.ravel(nz_left.sum(axis=1)).astype(bool)[self.P]
        self.closure_right_pde = nz_right_pde.astype(bool)[self.P]
        self.closure_right_jac = np.array(nz_right_jac.astype(bool).todense())[self.P, :]
        self.closure_ready = False
        partial_spsolve_pde_right(np.ones(3), np.arange(2, dtype=np.int32), np.arange(3, dtype=np.int32), np.zeros(2), np.zeros(2, dtype=np.int64))
        partial_spsolve_jac_right(np.ones(3), np.arange(2, dtype=np.int32), np.arange(3, dtype=np.int32), np.zeros((2, 2)), np.zeros((2, 1), dtype=np.int64), np.ones(2, dtype=np.int64))
        tmp_iter = np.zeros((2, ), dtype=np.int64)
        partial_spsolve_pde_left(np.ones(3), np.arange(2, dtype=np.int32), np.arange(3, dtype=np.int32), np.zeros(2), np.zeros(2, dtype=np.int64))
        partial_spsolve_jac_left(np.ones(3), np.arange(2, dtype=np.int32), np.arange(3, dtype=np.int32), np.zeros((2, 2)), np.flip(tmp_iter))

    def update(self, A):
        self.factor.cholesky_inplace(A)
        self.L = self.factor.L()
        self.L.eliminate_zeros()
        if not self.closure_ready:
            #time_start_all = perf_counter()
            #time_start = perf_counter()
            for j in range(self.n-1):
                row = self.L.indices[self.L.indptr[j]+1]
                self.closure_left[row] |= self.closure_left[j]
                self.closure_right_pde[row] |= self.closure_right_pde[j]
                self.closure_right_jac[row, :] |= self.closure_right_jac[j, :]
            #print(f'closures time elasped: {perf_counter() - time_start}')
            self.new_to_old_left = np.flatnonzero(self.closure_left)
#            self.old_to_new_left = self.closure_left * np.cumsum(self.closure_left) - 1
            self.new_to_old_right_pde = np.flatnonzero(self.closure_right_pde)
#            self.old_to_new_right_pde = self.closure_right_pde * np.cumsum(self.closure_right_pde) - 1
            new_to_old_right_jac_list = [np.flatnonzero(self.closure_right_jac[:, j]) for j in range(self.n)]
            self.new_to_old_right_jac_lengths = np.array([len(x) for x in new_to_old_right_jac_list])
            self.new_to_old_right_jac = np.zeros((self.n, np.max(self.new_to_old_right_jac_lengths)), dtype=int)
            for i in range(self.n):
                self.new_to_old_right_jac[i, :self.new_to_old_right_jac_lengths[i]] = new_to_old_right_jac_list[i]
#            self.left_to_right_pde = self.old_to_new_right_pde[self.new_to_old_left]
#            self.Pf = self.P[self.closure_right_pde]
#            self.Pb = self.old_to_new_left[np.argsort(self.P)][self.inz_left]
            self.Pt = np.argsort(self.P)
            self.Pto = self.Pt[self.inz_left]
            self.new_to_old_left_flipped = np.flip(self.new_to_old_left)
#            self.x_pde = np.zeros(self.new_to_old_right_pde.size + 1)
            self.x = np.zeros(self.n)
            self.X_jac = np.zeros((self.n, self.n))
#            self.sol = np.zeros(self.new_to_old_left.size)
            self.closure_ready = True
            #print(f'closure ready time elasped: {perf_counter() - time_start_all}')
        return self

        # def solve_pde_old(self, b):
        # time_start = perf_counter()
        # x = b[self.Pf]
        # print(f'permute time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # L_right = self.L[self.new_to_old_right_pde[:, None], self.new_to_old_right_pde[None, :]]
        # print(f'L_right time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # self.x_pde[:-1] = spl.spsolve(L_right, x)
        # print(f'forward solve time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # self.sol = self.x_pde[self.left_to_right_pde]
        # print(f'bridge time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # L_left = self.L[self.new_to_old_left[:, None], self.new_to_old_left[None, :]].T
        # print(f'L_left time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # solp = spl.spsolve(L_left, self.sol)
        # print(f'backward solve time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # sol = solp[self.Pb]
        # print(f'solution time elasped: {perf_counter() - time_start}')
        # return sol

    def solve_pde(self, b):
        self.x = b[self.P]
        #time_start = perf_counter()
        partial_spsolve_pde_right(self.L.data, self.L.indices, self.L.indptr, self.x, self.new_to_old_right_pde)
        #print(f'pde forward solve time elasped: {perf_counter() - time_start}')
        #time_start = perf_counter()
        partial_spsolve_pde_left(self.L.data, self.L.indices, self.L.indptr, self.x, np.arange(self.n-1, -1, -1))
        #print(f'pde backward solve time elasped: {perf_counter() - time_start}')
        return self.x[self.Pt]

    def solve_jac(self, B):
        #time_start = perf_counter()
        B[self.P, :].todense(out=self.X_jac)
        #print(f'permute time elasped: {perf_counter() - time_start}')
        #time_start = perf_counter()
        #print(f'{self.L.data.dtype}, {self.L.indices.dtype}, {self.L.indptr.dtype}, {self.X_jac.dtype}, {self.new_to_old_right_jac.dtype}, {self.new_to_old_right_jac_lengths.dtype}')
        partial_spsolve_jac_right(self.L.data, self.L.indices, self.L.indptr, self.X_jac, self.new_to_old_right_jac, self.new_to_old_right_jac_lengths)
        #print(f'forward solve time elasped: {perf_counter() - time_start}')
        #time_start = perf_counter()
        partial_spsolve_jac_left(self.L.data, self.L.indices, self.L.indptr, self.X_jac, self.new_to_old_left_flipped)
        #print(f'jac backward solve time elasped: {perf_counter() - time_start}')
        #time_start = perf_counter()
        return self.X_jac[self.Pto, :]
        #print(f'solution time elasped: {perf_counter() - time_start}')
        #return sol
        