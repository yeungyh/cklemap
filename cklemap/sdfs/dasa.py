import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import scipy.io as spio
from time import perf_counter


class DASAExp(object):

    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param):
        self.objfun = objfun
        self.solvefun = solvefun
        self.obj_sens_state = obj_sens_state
        self.obj_sens_param = obj_sens_param
        self.res_sens_state = res_sens_state
        self.res_sens_param = res_sens_param
        self.reset_timer()

    def reset_timer(self):
        self.solve_time = 0.0
        self.obj_time = 0.0
        self.dhdu_time = 0.0
        self.dhdp_time = 0.0
        self.dLdu_time = 0.0
        self.dLdp_time = 0.0
        self.adj_time = 0.0
        self.sens_time = 0.0
        self.grad_time = 0.0
        self.jac_time = 0.0

    def obj(self, p):
        time_start = perf_counter()
        self.u = self.solvefun(p)
        self.solve_time += perf_counter() - time_start
        time_start = perf_counter()
        obj = self.objfun(self.u, p)
        self.obj_time += perf_counter() - time_start
        return obj

    def grad(self, p):
        #u = self.solvefun(p)
        dhdu = self.obj_sens_state(self.u, p)
        dhdp = self.obj_sens_param(self.u, p)
        dLdu = self.res_sens_state(self.u, p)
        dLdp = self.res_sens_param(self.u, p)
        adj = -spl.spsolve(dLdu, dhdu)
        sens = dLdp.dot(adj)
        sens = sens + dhdp
        return sens


class DASAExpLM(DASAExp):

    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param, jac_size, top_size, init_jac=True):
        super().__init__(solvefun, objfun, obj_sens_state,
                         obj_sens_param, res_sens_state, res_sens_param)
        self.top_size = top_size
        self.jac = np.zeros(jac_size)
        if init_jac:
            self.jac[self.top_size:, :] = self.obj_sens_param(0, 0).todense()

    def grad(self, p):
        #u = self.solvefun(p)
        time_start_all = perf_counter()
        time_start = perf_counter()
        dhdu = self.obj_sens_state(self.u, p)
        self.dhdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdu = self.res_sens_state(self.u, p)
        self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj = spl.spsolve(dLdu, dhdu)
        sens = dLdp.T @ adj
        self.grad_time += perf_counter() - time_start
        self.jac[:self.top_size, :] = sens.T.todense()
        self.jac_time += perf_counter() - time_start_all
        return self.jac


class DASAExpLMAMPS(DASAExpLM):
    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param, state_sens_param, jac_size, top_size):
        super().__init__(solvefun, objfun, obj_sens_state, obj_sens_param,
                         res_sens_state, res_sens_param, jac_size, top_size)
        self.state_sens_param = state_sens_param

    def grad(self, p):
        time_start_all = perf_counter()
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        adj = self.state_sens_param()
        self.jac[:self.top_size, :] = adj.T @ dLdp
        self.grad_time += perf_counter() - time_start
        print(f'adj size = {adj.shape}, nnz = {np.count_nonzero(adj)}')
        self.jac_time += perf_counter() - time_start_all
        return self.jac


class DASAExpKL(DASAExpLM):

    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param, jac_size, top_size, const_term, param_sens_coeff, jacfun=None):
        super().__init__(solvefun, objfun, obj_sens_state, obj_sens_param,
                         res_sens_state, res_sens_param, jac_size, top_size, False)
        self.jacfun = jacfun
        self.const_term = const_term
        self.param_sens_coeff = param_sens_coeff.copy()
        self.jac[self.top_size:, :] = self.obj_sens_param(
            0, 0) @ self.param_sens_coeff

    def obj(self, xi):
        self.p = self.const_term + self.param_sens_coeff @ xi
        return super().obj(self.p)

    def grad(self, xi):
        time_start_all = perf_counter()
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, self.p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        time_start_grad = perf_counter()
        dhdu = self.obj_sens_state(self.u, self.p)
        self.dhdu_time += perf_counter() - time_start
        #time_start = perf_counter()
        dLdu = self.res_sens_state(self.u, self.p)
        #self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj = spl.spsolve(dLdu, dhdu)
        #adj = -self.jacfun(dhdu)
        self.adj_time += perf_counter() - time_start
        time_start = perf_counter()
        sens = adj.T @ dLdp
        self.sens_time += perf_counter() - time_start
        self.grad_time += perf_counter() - time_start_grad
        #print(f'grad time elasped: {perf_counter() - time_start_grad}')
        #self.jac[:self.top_size, :] = sens.todense() @ self.param_sens_coeff
        self.jac[:self.top_size, :] = sens @ self.param_sens_coeff
        self.jac_time += perf_counter() - time_start_all
        return self.jac


class DASAExpKLAMPS(DASAExpKL):
    def __init__(self, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param, state_sens_param, jac_size, top_size, const_term, param_sens_coeff):
        super().__init__(solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state,
                         res_sens_param, jac_size, top_size, const_term, param_sens_coeff)
        self.state_sens_param = state_sens_param

    def grad(self, xi):
        time_start_all = perf_counter()
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, self.p)
        self.dLdp_time += perf_counter() - time_start
#        print(f'shape: {dLdp.shape}, nnz: {dLdp.count_nonzero()}')
        time_start = perf_counter()
        adj = self.state_sens_param(dLdp)
        self.grad_time += perf_counter() - time_start
        #print(f'amps time elasped: {perf_counter() - time_start_all}')
#        sens = adj.T @ dLdp
#        self.jac[:self.top_size, :] = sens @ self.param_sens_coeff
        self.jac[:self.top_size, :] = adj @ self.param_sens_coeff
        self.jac_time += perf_counter() - time_start_all
        return self.jac


class DASAExpLMScalar(DASAExp):

    def grad(self, p):
        #u = self.solvefun(p)
        time_start = perf_counter()
        dhdu = self.obj_sens_state(self.u, p)
        self.dhdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dhdp = self.obj_sens_param(self.u, p)
        self.dhdp_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdu = self.res_sens_state(self.u, p)
        self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj = -spl.spsolve(dLdu, dhdu.T)
        self.adj_time += perf_counter() - time_start
        time_start = perf_counter()
        sens = dLdp.dot(adj)
        self.sens_time += perf_counter() - time_start
        return sens.T + dhdp


class DASAExpLMWithFlux(DASAExp):

    def __init__(self, NY, solvefun, objfun, obj_sens_state, obj_sens_param, res_sens_state, res_sens_param):
        super().__init__(solvefun, objfun, obj_sens_state,
                         obj_sens_param, res_sens_state, res_sens_param)
        self.NY = NY

    def obj(self, p):
        time_start = perf_counter()
        self.u = self.solvefun(p[:self.NY], p[self.NY:])
        self.solve_time += perf_counter() - time_start
        time_start = perf_counter()
        obj = self.objfun(self.u, p)
        self.obj_time += perf_counter() - time_start
        return obj

    def grad(self, p):
        Y = p[:self.NY]
        #q = p[self.NY:]
        #u = self.solvefun(Y, q)
        dhdu = self.obj_sens_state(self.u, Y)
        dhdp = self.obj_sens_param(self.u, p)
        dLdu = self.res_sens_state(self.u, Y)
        dLdp = self.res_sens_param(self.u, p)
        adj = -spl.spsolve((dLdu.T).tocsc(), dhdu.T)
        sens = dLdp.dot(adj)
        return sps.vstack([sens.T, dhdp]).toarray()
