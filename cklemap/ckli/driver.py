import numpy as np
import scipy.optimize as spo
from sdfs.geom import Geom
from sdfs.bc import BC
from sdfs.darcy import DarcyExp
from sdfs.dasa import DASAExpLM
from sdfs.tpfa import TPFA
from collections import namedtuple

import GPy
import ckli.ckliest as ckliest
import ckli.mapest as mapest
from time import perf_counter

def drive(rs, data, hp_from_data, Nx=32, Ny=32, ssv=None):

    # Geometry
    L = np.array([1.0, 1.0])
    N = np.array([Nx,  Ny])
    g = Geom(L, N)
    g.calculate()

    # Boundary conditions
    ul = 2.0
    ur = 1.0
    bc = BC(g)
    bc.dirichlet(g, "left", ul)
    bc.dirichlet(g, "right", ur)

    # Problem
    prob = DarcyExp(TPFA(g, bc), ssv)

    ## Reference fields
    
    # Reference log k field
    std_dev_ref = data['std_dev']
    cor_len_ref = data['cor_len']
    Nc = g.cells.num

    kernel = data['kernel']
    if kernel == 'se':
        gpy_kernel = GPy.kern.RBF
    elif kernel == 'exp':
        gpy_kernel = GPy.kern.Exponential
    elif kernel == 'm32':
        gpy_kernel = GPy.kern.Matern32
    elif kernel == 'm52':
        gpy_kernel = GPy.kern.Matern52
        
    se = gpy_kernel(input_dim=2, variance=std_dev_ref**2, lengthscale=cor_len_ref)
    CY = se.K(g.cells.centroids.T, g.cells.centroids.T)
    Yref = rs.multivariate_normal(np.zeros(Nc), CY)

    # Reference u field
    uref  = prob.solve(Yref)

    ## Measurements
    
    # Y masurements
    NYobs = data['NYobs']
    iYobs = rs.choice(Nc, NYobs, replace=False)
    Yobs  = Yref[iYobs]

    # u measurements
    Nuobs = data['Nuobs']
    iuobs = rs.choice(Nc, Nuobs, replace=False)
    uobs  = uref[iuobs]

    ## Y GP model
    if hp_from_data:
        print('Estimating hyperparameters from data...')
        sedd = GPy.kern.Matern52(input_dim=2, variance=std_dev_ref**2, lengthscale=cor_len_ref)
        mYref = GPy.models.GPRegression(g.cells.centroids[:,iYobs].T, Yobs[:,None], sedd, noise_var=1e-2)
        mYref.optimize(messages=True, ipython_notebook=False)
    else:
        mYref = GPy.models.GPRegression(g.cells.centroids[:,iYobs].T, Yobs[:,None], se, noise_var=np.sqrt(np.finfo(float).eps))

    Ypred, CYpred = mYref.predict_noiseless(g.cells.centroids.T, full_cov=True)
    Ypred = Ypred.flatten()

    ## u GP model
    ts = perf_counter()
    umean, Cu = ckliest.smc_gp(Ypred, CYpred, data['Nens'], prob, rs)
    upred, Cupred = ckliest.gpr(umean, Cu, uobs, iuobs)
    print(f'u GP: {perf_counter() - ts : g} s', flush=True)

    ## Eigendecomposition-based 

    NYxi = data['NYxi']
    Nuxi = data['Nuxi']
    PsiY, LambdaY = ckliest.KL_via_eigh(CYpred, NYxi)
    Psiu, Lambdau = ckliest.KL_via_eigh(Cupred, Nuxi)

    ## CKLI estimate

    res = ckliest.LeastSqRes(NYxi, Ypred, PsiY, Nuxi, upred, Psiu, prob, data['gamma_ckli'])
    x0  = np.zeros(Nuxi + NYxi)
    
    ts  = perf_counter()
    sol = spo.least_squares(res.val, x0, jac=res.jac, method='lm')
    ckli_status = sol.status
    print(f'CKLI: {perf_counter() - ts : g} s', flush=True)
    
    uxi = sol.x[:Nuxi]
    Yxi = sol.x[Nuxi:]
    uest = upred + Psiu.dot(uxi)
    Yest = Ypred + PsiY.dot(Yxi)
    print(f'CKLI optimality: {sol.optimality : g}')
    
    ## MAP H_1 estimate

    Lreg  = mapest.compute_Lreg(g)
    loss  = mapest.LossVec(Nc, Nc, iuobs, uobs, iYobs, Yobs, data['gamma_map'], Lreg) # H1 regularization
    dasa  = DASAExpLM(loss.val, loss.grad_u, loss.grad_Y, prob.solve, prob.residual_sens_u, prob.residual_sens_Y)
    
    ts    = perf_counter()
    sol   = spo.least_squares(dasa.obj, np.full(Nc, 0.0), jac=dasa.grad, method='lm')
    Yest_MAPH1 = sol.x
    MAP_status = sol.status
    print(f'MAP: {perf_counter() - ts : g} s', flush=True)

    output = namedtuple('output', ['CY', 'Yref', 'uref', 'Yest', 'uest', 'Yest_MAPH1', 'iYobs', 'iuobs', 'Yobs', 'uobs', 'ckli_status', 'MAP_status'])
    return output(CY=CY, Yref=Yref, uref=uref, Yest=Yest, uest=uest, Yest_MAPH1=Yest_MAPH1, iYobs=iYobs, iuobs=iuobs, Yobs=Yobs, uobs=uobs, ckli_status=ckli_status, MAP_status=MAP_status)
