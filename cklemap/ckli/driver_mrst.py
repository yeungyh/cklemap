import numpy as np
from scipy import spatial, optimize
import h5py
import hdf5storage
from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST
from sdfs.darcy import DarcyExp
from sdfs.dasa import DASAExpLM
from sdfs.tpfa import TPFA
from collections import namedtuple

import GPy
from sklearn import mixture
import ckli.ckliest as ckliest
import ckli.mapest as mapest
from time import perf_counter


def drive(rs, data, hp_from_data, geom, bc, ssv=None):

    # Problem
    prob = DarcyExp(TPFA(geom, bc), ssv)

    # Reference fields

    # Reference log k field
    Nc = geom.cells.num
    std_dev_ref = data['std_dev']
    cor_len_ref = data['cor_len']

    kernel = data['kernel']
    if kernel == 'se':
        gpy_kernel = GPy.kern.RBF
    elif kernel == 'exp':
        gpy_kernel = GPy.kern.Exponential
    elif kernel == 'm32':
        gpy_kernel = GPy.kern.Matern32
    elif kernel == 'm52':
        gpy_kernel = GPy.kern.Matern52

    se = gpy_kernel(input_dim=2, variance=std_dev_ref ** 2, lengthscale=cor_len_ref)
    CY = se.K(geom.cells.centroids.T, geom.cells.centroids.T)
    with h5py.File(data['conduct_filename'], 'r') as f:
        Yref = f.get('conduct_log')[:].ravel()

    tree = spatial.KDTree(geom.cells.centroids.T)
    Ymean = np.vectorize(lambda x: np.mean(Yref[x]))(
        tree.query_ball_point(geom.cells.centroids.T, data['mean_radius']))
    Yprime = Yref - Ymean

    # Reference u field
    uref = prob.solve(Yref)
    if False:
        hdf5storage.write({'head': uref}, filename='head.mat', matlab_compatible=True)

    # Measurements

    # Y masurements
    NYobs = data['NYobs']
    iYobs = rs.choice(Nc, NYobs, replace=False)
    Yobs = Yprime[iYobs]

    # u measurements
    Nuobs = data['Nuobs']
    iuobs = rs.choice(Nc, Nuobs, replace=False)
    uobs = uref[iuobs]

    dpgmm = mixture.BayesianGaussianMixture(n_components=30, covariance_type='full').fit(
        geom.cells.centroids.T, Yref)
    partition = dpgmm.predict(geom.cells.centroids.T)

    # Y GP model
    if hp_from_data:
        print('Estimating hyperparameters from data...')
        sedd = GPy.kern.Matern52(input_dim=2, variance=std_dev_ref**2, lengthscale=cor_len_ref)
        mYref = GPy.models.GPRegression(geom.cells.centroids[:, iYobs].T, Yobs[:, None], sedd, noise_var=1e-2)
        mYref.optimize(messages=True, ipython_notebook=False)
    else:
        mYref = GPy.models.GPRegression(geom.cells.centroids[:, iYobs].T, Yobs[:, None], se, noise_var=np.sqrt(np.finfo(float).eps))

    Ypred, CYpred = mYref.predict_noiseless(geom.cells.centroids.T, full_cov=True)
    Ypred = Ypred.ravel() + Ymean

    # u GP model
    ts = perf_counter()
    umean, Cu = ckliest.smc_gp(Ypred, CYpred, data['Nens'], prob.solve, rs)
    upred, Cupred = ckliest.gpr(umean, Cu, uobs, iuobs)
    print(f'u GP: {perf_counter() - ts : g} s', flush=True)

    # Eigendecomposition-based

    NYxi = data['NYxi']
    Nuxi = data['Nuxi']
    PsiY, LambdaY = ckliest.KL_via_eigh(CYpred, NYxi)
    Psiu, Lambdau = ckliest.KL_via_eigh(Cupred, Nuxi)

    # CKLI estimate

    res = ckliest.LeastSqRes(NYxi, Ypred, PsiY, Nuxi, upred, Psiu, prob, data['gamma_ckli'])
    x0 = np.zeros(Nuxi + NYxi)

    ts = perf_counter()
    sol = optimize.least_squares(res.val, x0, jac=res.jac, method='lm')
    ckli_status = sol.status
    print(f'CKLI: {perf_counter() - ts : g} s', flush=True)

    uxi = sol.x[:Nuxi]
    Yxi = sol.x[Nuxi:]
    uest = upred + Psiu.dot(uxi)
    Yest = Ypred + PsiY.dot(Yxi)
    print(f'CKLI optimality: {sol.optimality : g}')

    # MAP H_1 estimate

    Lreg = mapest.compute_Lreg(geom)
    # H1 regularization
    loss = mapest.LossVec(iuobs, uobs, iYobs, Yref[iYobs], data['gamma_map'], Lreg)
    dasa = DASAExpLM(loss.val, loss.grad_u, loss.grad_Y, prob.solve, prob.residual_sens_u, prob.residual_sens_Y)

    ts = perf_counter()
    sol = optimize.least_squares(
        dasa.obj, np.zeros(Nc), jac=dasa.grad, method='lm')
    Yest_MAPH1 = sol.x
    MAP_status = sol.status
    print(f'MAP: {perf_counter() - ts : g} s', flush=True)
    uest_MAPH1 = prob.solve(Yest_MAPH1)

    output = namedtuple('output', ['CY', 'partition', 'Ymean', 'Yref', 'uref', 'Ypred', 'upred', 'Yest',
                                   'uest', 'Yest_MAPH1', 'uest_MAPH1', 'iYobs', 'iuobs', 'Yobs', 'uobs',
                                   'ckli_status', 'MAP_status'])
    return output(CY=CY, partition=partition, Ymean=Ymean, Yref=Yref, uref=uref, Ypred=Ypred, upred=upred,
                  Yest=Yest, uest=uest, Yest_MAPH1=Yest_MAPH1, uest_MAPH1=uest_MAPH1, iYobs=iYobs,
                  iuobs=iuobs, Yobs=Yobs, uobs=uobs, ckli_status=ckli_status, MAP_status=MAP_status)
