from time import perf_counter
import numpy as np
import scipy.linalg as spl
# from joblib import Parallel, delayed


def gpr(ymean, Cy, yobs, iobs):
    Cytest = Cy[iobs]
    L = spl.cholesky(Cy[np.ix_(iobs, iobs)] +
                     np.sqrt(np.finfo(float).eps) * np.eye(iobs.size), lower=True)
    a = spl.solve_triangular(L.T, spl.solve_triangular(
        L, yobs - ymean[iobs], lower=True))
    V = spl.solve_triangular(L, Cytest, lower=True)
    return ymean + Cytest.T @ a, Cy - V.T @ V


def smc_gp(Ypred, CYpred, Nens, prob, rs, verbose=False, randomize_bc=False, randomize_scale=0.01):
    Nc = Ypred.size

    timer = perf_counter()

    Lpred = spl.cholesky(
        CYpred + np.sqrt(np.finfo(float).eps) * np.eye(Nc), lower=True)
    if randomize_bc:
        uens = np.vstack([prob.randomize_bc('N', randomize_scale).solve(
            Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)])
    else:
        uens = np.vstack([prob.solve(Ypred + Lpred @ rs.randn(Nc))
                         for _ in range(Nens)])
        # with Parallel(n_jobs=8) as parallel:
        #    uens = np.vstack(parallel(delayed(prob.solve)(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)))

    if verbose:
        print(f'Elapsed time: {perf_counter() - timer : g} s')

    return np.mean(uens, axis=0), np.cov(uens, rowvar=False, bias=False)


def KL_via_eigh(C, Nxi):
    Nc = C.shape[0]
    Lambda, Phi = spl.eigh(C, eigvals=(Nc - Nxi, Nc - 1))
    return (Phi.real @ np.diag(np.sqrt(np.abs(Lambda))))[:, ::-1], Lambda[::-1]


class LeastSqRes(object):

    def __init__(self, NYxi, Ypred, PsiY, Nuxi, upred, Psiu, problem, Ygamma, ugamma, res_fac, L, iuobs, uobs, iYobs, Yobs, beta, ssv=None):
        self.NYxi = NYxi
        self.Nuxi = Nuxi
        self.Nc = Ypred.shape[0] if ssv is None else ssv.size
        self.problem = problem
        self.Ypred = Ypred
        self.PsiY = PsiY
        self.upred = upred
        self.Psiu = Psiu
        self.beta12 = np.sqrt(beta)
        self.LYgamma12 = np.sqrt(Ygamma) * L
        self.Lugamma12 = np.sqrt(ugamma) * L
        self.res_fac = res_fac
        self.L = L
        self.ssv = ssv
        self.iuobs = iuobs
        self.uobs = uobs
        self.iYobs = iYobs
        self.Yobs = Yobs
        self.jac_mat = np.block([[np.zeros((self.Nc, self.Nuxi + self.NYxi))],
                                 [-self.beta12 * self.Psiu[self.iuobs],
                                  np.zeros((np.size(self.iuobs), self.NYxi))],
                                 [self.Lugamma12 @ self.Psiu,
                                  np.zeros((self.L.shape[0], self.NYxi))],
                                 [np.zeros((self.L.shape[0], self.Nuxi)),
                                  self.LYgamma12 @ self.PsiY]])

    def val(self, x):
        self.u = self.upred + self.Psiu @ x[:self.Nuxi]
        self.Y = self.Ypred + self.PsiY @ x[self.Nuxi:]
        res = self.problem.residual(self.u, self.Y)
        if self.ssv is not None:
            res = res[self.ssv]
        return np.concatenate((res / self.res_fac, self.beta12 * (self.uobs - self.u[self.iuobs]), self.Lugamma12 @ self.u, self.LYgamma12 @ self.Y))

    def objectives(self, u, Y):
        return self.problem.residual(u, Y) / self.res_fac, self.beta12 * (self.uobs - u[self.iuobs]), self.Lugamma12 @ u, self.LYgamma12 @ Y

    def jac(self, _x):
        del _x
        au = self.problem.residual_sens_u(self.u, self.Y) / self.res_fac
        aY = self.problem.residual_sens_Y(self.u, self.Y).T / self.res_fac
        if self.ssv is not None:
            au = au[self.ssv]
            aY = aY[self.ssv]
        self.jac_mat[:self.Nc, :self.Nuxi] = au @ self.Psiu
        self.jac_mat[:self.Nc, self.Nuxi:] = aY @ self.PsiY
        return self.jac_mat
