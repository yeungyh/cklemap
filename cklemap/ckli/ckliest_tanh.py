from time import perf_counter
import numpy as np
import scipy.linalg as spl
from joblib import Parallel, delayed

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
        uens = np.vstack([prob.randomize_bc('N', randomize_scale).solve(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)])
    else:
        uens = np.vstack([prob.solve(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)])
        #with Parallel(n_jobs=8) as parallel:
        #    uens = np.vstack(parallel(delayed(prob.solve)(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)))

    if verbose:
        print(f'Elapsed time: {perf_counter() - timer : g} s')

    return np.mean(uens, axis=0), np.cov(uens, rowvar=False, bias=False)

def KL_via_eigh(C, Nxi):
    Nc = C.shape[0]
    Lambda, Phi = spl.eigh(C, eigvals=(Nc - Nxi, Nc - 1))
    return (Phi.real @ np.diag(np.sqrt(np.abs(Lambda))))[:, ::-1], Lambda[::-1]

class LeastSqRes(object):

    def __init__(self, NYxi, Ypred, PsiY, Nuxi, upred, Psiu, problem, Ygamma, ugamma, iuobs, uobs, iYobs, Yobs, beta, ssv=None):
        self.NYxi     = NYxi
        self.Nuxi     = Nuxi
        self.problem  = problem
        self.Ypred    = Ypred
        self.PsiY     = PsiY
        self.upred    = upred
        self.Psiu     = Psiu
        self.beta12   = np.sqrt(beta)
        self.Ygamma12 = np.sqrt(Ygamma)
        self.ugamma12 = np.sqrt(ugamma)
        self.ssv      = ssv
        self.iuobs    = iuobs
        self.uobs     = uobs
        self.iYobs    = iYobs
        self.Yobs     = Yobs
        self.jconst   = np.block([[-self.beta12 * self.Psiu[self.iuobs], np.zeros((np.size(self.iuobs), self.NYxi))],
                                  [np.zeros((np.size(self.iYobs), self.Nuxi)), -self.beta12 * self.PsiY[self.iYobs]]])

    def val(self, x):
        uxi = x[:self.Nuxi]
        Yxi = x[self.Nuxi:]
        self.u = self.upred + self.Psiu.dot(np.tanh(uxi))
        self.Y = self.Ypred + self.PsiY.dot(np.tanh(Yxi))
        residual = self.problem.residual(self.u, self.Y)
        if self.ssv is None:
            ret = np.concatenate((residual, self.beta12 * (self.uobs - self.u[self.iuobs]), self.beta12 * (self.Yobs - self.Y[self.iYobs])))
        else:
            ret = np.concatenate((self.problem.residual(self.u, self.Y)[self.ssv], self.beta12 * (self.uobs - self.u[self.iuobs]), self.beta12 * (self.Yobs - self.Y[self.iYobs])))
        print(f'{residual.shape=}, {ret.shape=}, {x.shape=}')
        return ret
            
    def jac(self, x):
        uxi = x[:self.Nuxi]
        Yxi = x[self.Nuxi:]
        au = self.problem.residual_sens_u(self.u, self.Y)
        aY = self.problem.residual_sens_Y(self.u, self.Y)
        if self.ssv is None:
            return np.block([[au.dot(self.Psiu * (1 - np.tanh(uxi) ** 2)), aY.T.dot(self.PsiY * (1 - np.tanh(Yxi) ** 2))], [self.jconst]])
        else:
            return np.block([[au.dot(self.Psiu * (1 - np.tanh(uxi) ** 2))[self.ssv,:], aY.T.dot(self.PsiY * (1 - np.tanh(Yxi) ** 2))[self.ssv,:]], [self.jconst]])