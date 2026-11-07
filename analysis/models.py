"""Model definitions. All preprocessing lives inside the pipelines, so it is refit in every fold."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.dummy import DummyRegressor


def gpr(ard_dims=None):
    """GPR with Matern (nu=1.5) kernel and a white-noise term; ARD when ard_dims is given."""
    ls = np.ones(ard_dims) if ard_dims else 1.0
    kernel = C(1.0, (1e-2, 1e2)) * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=1.5) \
        + WhiteKernel(0.1, (1e-5, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)


def make(name):
    if name == "mean":
        return DummyRegressor()
    if name == "kinematic":          # Ra = a + b * f^2 (column 1 of PAR is the feed)
        return make_pipeline(FunctionTransformer(lambda X: X[:, [1]] ** 2), LinearRegression())
    if name == "param_lin":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "param_gpr":
        return make_pipeline(StandardScaler(), gpr(3))
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    if name.startswith("gpr_k"):     # in-fold selection of the k best features (F-test)
        k = int(name[5:])
        return make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), gpr())
    raise ValueError(name)
