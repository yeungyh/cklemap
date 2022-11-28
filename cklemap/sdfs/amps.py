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
def partial_spsolve_pde(L, iL, jL, b, cls_right):
    for j in cls_right:
        b[j] /= L[jL[j]]
        b[iL[jL[j]+1:jL[j+1]]] -= L[jL[j]+1:jL[j+1]] * b[j]

    for j in range(b.shape[0]-1, -1, -1):
        b[j] -= np.dot(L[jL[j]+1:jL[j+1]], b[iL[jL[j]+1:jL[j+1]]])
        b[j] /= L[jL[j]]


# def partial_spsolve_jac(L, iL, jL, B, C, cls_right_i, cls_right_j, cls_left_i, cls_left_j):
@njit(parallel=True)
def partial_spsolve_jac(L, iL, jL, B, cls_right_i, cls_right_j):
    for i in prange(B.shape[1]):
        for j in cls_right_i[cls_right_j[i]:cls_right_j[i+1]]:
            B[j, i] /= L[jL[j]]
            B[iL[jL[j]+1:jL[j+1]], i] -= L[jL[j]+1:jL[j+1]] * B[j, i]

    # for i in prange(C.shape[1]):
    #     for j in cls_left_i[cls_left_j[i]:cls_left_j[i+1]]:
    #         C[j, i] /= L[jL[j]]
    #         C[iL[jL[j]+1:jL[j+1]], i] -= L[jL[j]+1:jL[j+1]] * C[j, i]

    for j in range(B.shape[0]-1, -1, -1):
        B[j] -= L[jL[j]+1:jL[j+1]] @ B[iL[jL[j]+1:jL[j+1]]]
        B[j] /= L[jL[j]]


class AMPS(object):

    def __init__(self, A, nz_right_pde, iuobs):
        self.factor = cm.analyze(
            A, mode="auto", ordering_method="nesdis", use_long=False)
        self.n = A.shape[0]
        self.P = self.factor.P()
        self.Pt = np.argsort(self.P)
        self.L = None
        self.x = np.zeros(self.n)
        #self.X_jac = nz_right_jac.todense()[self.P, :]
        self.X_jac_right = np.zeros((self.n, iuobs.size))
        self.X_jac_right[self.Pt[iuobs], np.arange(iuobs.size)] = 1
#        self.inz_left = np.ravel(nz_left.sum(axis=1)).astype(bool)
#        self.closure_left_jac = nz_left_jac.astype(
#            bool).toarray(order='F')[self.P, :]
        self.closure_right_pde = nz_right_pde.astype(bool)[self.P]
        self.closure_right_jac = self.X_jac_right.astype(bool, copy=True)
        self.closure_ready = False
        partial_spsolve_pde(np.ones(3), np.arange(2, dtype=np.int32), np.arange(
            3, dtype=np.int32), np.ones(2), np.zeros(2, dtype=np.int64))
        partial_spsolve_jac(np.ones(3), np.arange(2, dtype=np.int32), np.arange(3, dtype=np.int32), np.ones(
            (2, 2)), np.zeros(2, dtype=np.int64), np.arange(3, dtype=np.int64))
        # partial_spsolve_jac(np.ones(3), np.arange(2, dtype=np.int32), np.arange(3, dtype=np.int32), np.ones(
        #     (2, 2)), np.ones((2, 2)), np.zeros(2, dtype=np.int64), np.arange(3, dtype=np.int64), np.zeros(2, dtype=np.int64), np.arange(3, dtype=np.int64))

    def update(self, A):
        self.factor.cholesky_inplace(A)
        self.L = self.factor.L()
        # print(f'factor update time elasped: {perf_counter() - time_start}')
        if not self.closure_ready:
            # time_start = perf_counter()
            for j in range(self.n-1):
                row = self.L.indices[self.L.indptr[j]+1]
#                self.closure_left_jac[row] |= self.closure_left_jac[j]
                self.closure_right_pde[row] |= self.closure_right_pde[j]
                self.closure_right_jac[row] |= self.closure_right_jac[j]
            # print(f'closures time elasped: {perf_counter() - time_start}')
#            self.new_to_old_left_jac = np.flatnonzero(self.closure_left_jac)
#            self.old_to_new_left = self.closure_left * np.cumsum(self.closure_left) - 1
            self.new_to_old_right_pde = np.flatnonzero(self.closure_right_pde)
#            self.old_to_new_right_pde = self.closure_right_pde * np.cumsum(self.closure_right_pde) - 1
#            nonzeros_right_jac = self.closure_right_jac.nonzero()
#            nonzeros_right_jac_rows, nonzeros_right_jac_indices, self.nonzeros_right_jac_lengths = np.unique(
#                nonzeros_right_jac[0], return_index=True, return_counts=True)
#            self.closure_left_jac_csc = sps.csc_matrix(self.closure_left_jac)
            self.closure_right_jac_csc = sps.csc_matrix(self.closure_right_jac)
#            print(f'{self.closure_left_jac_csc.shape=}')
#            np.savetxt('closure_left_nnz.txt',
#                       self.closure_left_jac_csc.getnnz(axis=1).astype(int), fmt='%i')
#            self.left_to_right_pde = self.old_to_new_right_pde[self.new_to_old_left]
#            self.Pf = self.P[self.closure_right_pde]
#            self.Pb = self.old_to_new_left[np.argsort(self.P)][self.inz_left]

#            self.Pto = self.Pt[self.inz_left]
#            self.new_to_old_left_flipped = np.flip(self.new_to_old_left)
#            self.x_pde = np.zeros(self.new_to_old_right_pde.size + 1)
#            self.sol = np.zeros(self.new_to_old_left.size)
            self.closure_ready = True
            # print(f'closure ready time elasped: {perf_counter() - time_start_all}')
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
        # time_start = perf_counter()
        partial_spsolve_pde(self.L.data, self.L.indices, self.L.indptr,
                            self.x, self.new_to_old_right_pde)
        # print(f'pde solve time elasped: {perf_counter() - time_start}')
        return self.x[self.Pt]

    def solve_jac(self):
        X_jac = self.X_jac_right.copy()
        # Z_jac = premultiplier.toarray(order='F')[self.P, :]
        #X_jac = B[self.P, :].todense()
        # print(f'permute time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # print(f'{self.L.data.dtype}, {self.L.indices.dtype}, {self.L.indptr.dtype}, {self.X_jac.dtype}, {self.new_to_old_right_jac.dtype}, {self.new_to_old_right_jac_lengths.dtype}')
        # print(f'forward solve time elasped: {perf_counter() - time_start}')
        # time_start=perf_counter()
        # partial_spsolve_jac(self.L.data, self.L.indices, self.L.indptr, X_jac, Z_jac,
        #                     self.closure_right_jac_csc.indices, self.closure_right_jac_csc.indptr,
        #                     self.closure_left_jac_csc.indices, self.closure_left_jac_csc.indptr)
        partial_spsolve_jac(self.L.data, self.L.indices, self.L.indptr, X_jac,
                            self.closure_right_jac_csc.indices, self.closure_right_jac_csc.indptr)
        return X_jac
        # return X_jac.T @ Z_jac
        # print(f'jac solve time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # return X_jac[self.Pt]
        # print(f'solution time elasped: {perf_counter() - time_start}')
        # return sol


@njit
def spsolve_pde(L, iL, jL, b, left_range, right_range):
    for j in right_range:
        b[j] /= L[jL[j]]
        b[iL[jL[j]+1:jL[j+1]]] -= L[jL[j]+1:jL[j+1]] * b[j]

    for j in left_range:
        b[j] -= np.dot(L[jL[j]+1:jL[j+1]], b[iL[jL[j]+1:jL[j+1]]])
        b[j] /= L[jL[j]]


@njit(parallel=True)
def spsolve_jac(L, iL, jL, B, left_range, right_range):
    for i in prange(B.shape[1]):
        for j in right_range[i]:
            B[j, i] /= L[jL[j]]
            B[iL[jL[j]+1:jL[j+1]], i] -= L[jL[j]+1:jL[j+1]] * B[j, i]

    for j in left_range:
        B[j, :] -= L[jL[j]+1:jL[j+1]] @ B[iL[jL[j]+1:jL[j+1]], :]
        B[j, :] /= L[jL[j]]


class AMPS_full(object):

    def __init__(self, A, Nuobs):
        self.factor = cm.analyze(
            A, mode="auto", ordering_method="nesdis", use_long=False)
        self.n = A.shape[0]
        self.Nuobs = Nuobs
        self.P = self.factor.P()
        self.Pt = np.argsort(self.P)
        self.L = None
        self.x = np.zeros(self.n)
        self.X_jac = np.zeros((self.n, self.Nuobs))
        spsolve_pde(np.ones(3), np.arange(2, dtype=np.int32), np.arange(
            3, dtype=np.int32), np.zeros(2), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64))
        spsolve_jac(np.ones(3), np.arange(2, dtype=np.int32), np.arange(
            3, dtype=np.int32), np.zeros((2, 2)), np.zeros((2, ), dtype=np.int64), np.zeros((2, 1), dtype=np.int64))

    def update(self, A):
        time_start = perf_counter()
        self.factor.cholesky_inplace(A)
        self.L = self.factor.L()
        # print(f'factor update time elasped: {perf_counter() - time_start}')
        return self

    def solve_pde(self, b):
        self.x = b[self.P]
        time_start = perf_counter()
        spsolve_pde(self.L.data, self.L.indices, self.L.indptr,
                    self.x, np.arange(self.n-1, -1, -1), np.arange(self.n))
        # print(f'pde solve time elasped: {perf_counter() - time_start}')
        return self.x[self.Pt]

    def solve_jac(self, B):
        # time_start = perf_counter()
        B[self.P, :].todense(out=self.X_jac)
        # print(f'permute time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        # print(f'{self.L.data.dtype}, {self.L.indices.dtype}, {self.L.indptr.dtype}, {self.X_jac.dtype}, {self.new_to_old_right_jac.dtype}, {self.new_to_old_right_jac_lengths.dtype}')
        # print(f'forward solve time elasped: {perf_counter() - time_start}')
        time_start = perf_counter()
        spsolve_jac(self.L.data, self.L.indices, self.L.indptr,
                    self.X_jac, np.arange(self.n-1, -1, -1), np.tile(np.arange(self.n), (self.Nuobs, 1)))
        # print(f'jac solve time elasped: {perf_counter() - time_start}')
        # time_start = perf_counter()
        return self.X_jac[self.Pt, :]
        # print(f'solution time elasped: {perf_counter() - time_start}')
        # return sol
