from time import perf_counter
import numpy as np
import scipy.linalg as spl

def gpr(ymean, Cy, yobs, iobs):
    Cytest = Cy[iobs]
    L = spl.cholesky(Cy[np.ix_(iobs, iobs)] + np.sqrt(np.finfo(float).eps) * np.eye(iobs.size), lower=True)
    a = spl.solve_triangular(L.T, spl.solve_triangular(L, yobs - ymean[iobs], lower=True))
    V = spl.solve_triangular(L, Cytest, lower=True)
    return ymean + Cytest.T @ a, Cy - V.T @ V

def smc_gp(Ypred, CYpred, Nens, prob, rs, verbose=False, randomize_bc=False, randomize_scale=0.01):
    Nc = Ypred.size
    
    timer = perf_counter()

    Lpred = spl.cholesky(CYpred + np.sqrt(np.finfo(float).eps) * np.eye(Nc), lower=True)
    if randomize_bc:
        uens = np.vstack([prob.randomize_bc('N').solve(Ypred + Lpred.dot(rs.randn(Nc))) for _ in range(Nens)])
    else:
        uens = np.vstack([prob.solve(Ypred + Lpred.dot(rs.randn(Nc))) for _ in range(Nens)])

    if verbose:
        print(f'Elapsed time: {perf_counter() - timer : g} s')

    return np.mean(uens, axis=0), np.cov(uens, rowvar=False, bias=False)

def KL_via_eigh(C, Nxi):
    Nc = C.shape[0]
    Lambda, Phi = spl.eigh(C, eigvals=(Nc - Nxi, Nc - 1))
    return (Phi.real @ np.diag(np.sqrt(np.abs(Lambda))))[:, ::-1], Lambda[::-1]

class LeastSqRes(object):

    def __init__(self, NYxi, Ypred, PsiY, Nuxi, upred, Psiu, problem, gamma, ssv=None):
        self.NYxi    = NYxi
        self.Nuxi    = Nuxi
        self.problem = problem
        self.Ypred   = Ypred
        self.PsiY    = PsiY
        self.upred   = upred
        self.Psiu    = Psiu
        self.gamma12 = np.sqrt(gamma)
        self.ssv     = ssv
        self.eye_g12 = self.gamma12 * np.eye(self.Nuxi + self.NYxi)

    def val(self, x):
        uxi = x[:self.Nuxi]
        Yxi = x[self.Nuxi:]
        u = self.upred + self.Psiu.dot(uxi)
        Y = self.Ypred + self.PsiY.dot(Yxi)
        # return self.problem.residual(u, Y)
        #if self.ssv is None:
        return np.concatenate((self.problem.residual(u, Y), self.gamma12 * x))
        #else:
        #    return np.concatenate((self.problem.residual(u, Y)[self.ssv], self.gamma12 * uxi, self.gamma12 * Yxi))
            
    def jac(self, x):
        uxi = x[:self.Nuxi]
        Yxi = x[self.Nuxi:]
        u = self.upred + self.Psiu.dot(uxi)
        Y = self.Ypred + self.PsiY.dot(Yxi)
        au = self.problem.residual_sens_u(u, Y)
        aY = self.problem.residual_sens_Y(u, Y)
        #if self.ssv is None:
        return np.block([[au.dot(self.Psiu), aY.T.dot(self.PsiY)], [self.eye_g12]])
        #else:
        #    return np.concatenate((jacr[self.ssv,:], self.gamma12 * np.eye(self.Nuxi + self.NYxi)))