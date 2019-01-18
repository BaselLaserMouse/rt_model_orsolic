import operator
from functools import partial, reduce
from itertools import product

import gpflow
import numpy as np
import tensorflow as tf
from scipy.special import expit, logsumexp
from sklearn.cluster import MiniBatchKMeans
from tensorflow.contrib.distributions import Normal, Gamma

from gp_advi import FVGP, build_factor


def _prepare_X(dset_row, n_lags, max_nt):
    """convert stimulus data into lagged version (helper function)"""
    nt = len(dset_row.ys)
    X = np.zeros((nt, n_lags + 3))
    for i in range(min(n_lags, nt)):
        X[i:, i] = dset_row.ys[:nt-i]
    X[:, -3] = np.arange(nt) / max_nt
    X[:, -2] = dset_row.hazard_code
    X[:, -1] = dset_row.mouse_code
    return X


def prepare_X(dset, n_lags, max_nt):
    """convert stimulus data into lagged version"""
    return [
        _prepare_X(dset_row, n_lags, max_nt) for _, dset_row in dset.iterrows()
    ]


def prepare_Xy(dset, n_lags, max_nt):
    """convert stimuli/reaction-time into lagged data and binary responses"""
    Xs, ys = [], []

    for _, row in dset.iterrows():
        X = _prepare_X(row, n_lags, max_nt)
        y = np.zeros((len(X), 1))

        if not np.isnan(row.rt):
            last_idx = int(row.rt) + 1
            X, y = X[:last_idx], y[:last_idx]
            y[-1] = 1

        Xs.append(X)
        ys.append(y)

    return Xs, ys


class ProjKernel(gpflow.kernels.Kernel):

    def __init__(self, base_kernel, input_dim, K, W=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        # disable lengthscales optimisation for sub-kernels
        kernel_stack = [base_kernel]
        while kernel_stack:
            kernel = kernel_stack.pop()
            if isinstance(kernel, gpflow.kernels.Stationary):
                kernel.lengthscales.trainable = False
            elif isinstance(kernel, gpflow.kernels.Combination):
                kernel_stack.extend(kernel.kernels)

        self.base_kernel = base_kernel

        if W is None:
            W = np.random.randn(input_dim, K)
        self.W = gpflow.params.Parameter(W)

    def _project(self, X, X2=None):
        X_sliced, X2_sliced = self._slice(X, X2)
        X = tf.concat([X, tf.matmul(X_sliced, self.W)], 1)
        if X2 is not None:
            X2 = tf.concat([X2, tf.matmul(X2_sliced, self.W)], 1)
        return X, X2

    def Kdiag(self, X):
        X, _ = self._project(X, None)
        return self.base_kernel.Kdiag(X)

    def K(self, X, X2=None):
        X, X2 = self._project(X, X2)
        return self.base_kernel.K(X, X2)


def _warping(X, a, b, c):
    return X + tf.reduce_sum(a * tf.tanh(b * X[..., np.newaxis] + c), axis=-1)


class WarpedKernel(gpflow.kernels.Kernel):

    def __init__(self, base_kernel, n_tanh):
        super().__init__(base_kernel.input_dim, active_dims=None)
        self.base_kernel = base_kernel

        coeffs_a = np.abs(np.random.randn(n_tanh))
        coeffs_b = np.abs(np.random.randn(n_tanh))
        coeffs_c = np.random.randn(n_tanh)

        self.coeffs_a = gpflow.params.Parameter(
            coeffs_a, gpflow.transforms.Exp()
        )
        self.coeffs_b = gpflow.params.Parameter(
            coeffs_b, gpflow.transforms.Exp()
        )
        self.coeffs_c = gpflow.params.Parameter(coeffs_c)

    def _warp(self, X, X2=None):
        X_sliced, X2_sliced = self._slice(X, X2)
        X = _warping(X, self.coeffs_a, self.coeffs_b, self.coeffs_c)
        if X2 is not None:
            X2 = _warping(X2, self.coeffs_a, self.coeffs_b, self.coeffs_c)
        return X, X2

    def Kdiag(self, X):
        X, _ = self._warp(X, None)
        return self.base_kernel.Kdiag(X)

    def K(self, X, X2=None):
        X, X2 = self._warp(X, X2)
        return self.base_kernel.K(X, X2)


class LogTranformedKernel(gpflow.kernels.Kernel):

    def __init__(self, base_kernel):
        super().__init__(base_kernel.input_dim, active_dims=None)
        self.base_kernel = base_kernel
        self.offset = gpflow.params.Parameter(1.0, gpflow.transforms.Exp())

    def _warp(self, X, X2=None):
        X_sliced, X2_sliced = self._slice(X, X2)
        X = tf.log(X + self.offset)
        if X2 is not None:
            X2 = tf.log(X2 + self.offset)
        return X, X2

    def Kdiag(self, X):
        X, _ = self._warp(X, None)
        return self.base_kernel.Kdiag(X)

    def K(self, X, X2=None):
        X, X2 = self._warp(X, X2)
        return self.base_kernel.K(X, X2)


class ExpProjKernel(gpflow.kernels.Kernel):

    def __init__(self, base_kernel, input_dim, n_exp,
                 ND=None, A=None, L=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        # disable lengthscales optimisation for sub-kernels
        kernel_stack = [base_kernel]
        while kernel_stack:
            kernel = kernel_stack.pop()
            if isinstance(kernel, gpflow.kernels.Stationary):
                kernel.lengthscales.trainable = False
            elif isinstance(kernel, gpflow.kernels.Combination):
                kernel_stack.extend(kernel.kernels)

        self.base_kernel = base_kernel

        if A is None:
            A = np.random.randn(1, n_exp)
        if ND is None:
            ND = 0.0
        if L is None:
            L = np.ones([1, n_exp])

        self.A = gpflow.params.Parameter(A)
        self.ND = gpflow.params.Parameter(ND)
        self.L = gpflow.params.Parameter(L)
        self.input_dim = input_dim
        self.n_exp = n_exp

    @gpflow.params_as_tensors
    def _makeW(self):
        t = tf.expand_dims(tf.range(self.input_dim, dtype='float64'), 1)
        # make sigmoids for non-decision time
        nd = tf.reciprocal(tf.exp(tf.negative(t)+self.ND)+1.0)
        nd = tf.tile(nd, [1, self.n_exp])
        a = tf.tile(self.A, [self.input_dim, 1])
        # multiply exponential with a sigmoid
        W = tf.multiply(
                tf.multiply(tf.exp(tf.matmul(t, self.L)), a),
                nd
                )
        return W

    def _project(self, X, X2=None):
        X_sliced, X2_sliced = self._slice(X, X2)
        W = self._makeW()
        X = tf.concat([X, tf.matmul(X_sliced, W)], 1)
        if X2 is not None:
            X2 = tf.concat([X2, tf.matmul(X2_sliced, W)], 1)
        return X, X2

    def Kdiag(self, X):
        X, _ = self._project(X, None)
        return self.base_kernel.Kdiag(X)

    def K(self, X, X2=None):
        X, X2 = self._project(X, X2)
        return self.base_kernel.K(X, X2)


# taken from https://github.com/GPflow/GPflow/issues/331#issuecomment-277683687
class Hierarchical(gpflow.kernels.Combination):
    """
    Kernel for hierarchical models

    Hensman et al 2013, "Hierarchical Bayesian modelling of gene expression
    time series across irregularly sampled replicates and clusters"
    http://www.biomedcentral.com/1471-2105/14/252

    The active_dims argument to the underlying kernels inform the kernel which
    columns of X to use to calculate covariance.

    The list indicator_dims tells the kernel which columns of X to use for
    indicating batches.

    Example initialization:

        k = Hierarchical([gpflow.kernels.RBF(1, active_dims=[0]),
                          gpflow.kernels.RBF(1, active_dims=[0])],
                         indicator_dims=[1])
    """
    def __init__(self, kernels, indicator_dims=[1]):
        gpflow.kernels.Combination.__init__(self, kernels)
        self.indicator_dims = indicator_dims

    def K(self, X, X2=None):
        K = self.kernels[0].K(X, X2)

        if X2 is None:
            X2 = X

        for i, ind_dim in enumerate(self.indicator_dims):
            indX, indX2 = X[:, ind_dim:ind_dim + 1], X2[:, ind_dim:ind_dim + 1]
            mask = tf.to_double(tf.equal(indX, tf.transpose(indX2)))
            k = self.kernels[i + 1]
            K += mask * k.K(X, X2)

        return K

    def Kdiag(self, X):
        K = self.kernels[0].Kdiag(X)
        for k in self.kernels[1:]:
            K += k.Kdiag(X)
        return K


def build_kernel(kernel_type, kernel_input, hierarchy, n_lags, n_proj=None,
                 sigma=None, n_tanh=None):
    """construct the GP kernel to fit reaction-time"""

    # retrieve a class corresponding to the kernel type
    kernel_class = getattr(gpflow.kernels, str(kernel_type))

    if kernel_class is gpflow.kernels.Linear:
        kernel_class = partial(kernel_class, variance=0.01)

    elif kernel_class is gpflow.kernels.White:
        kernel_class = partial(kernel_class, variance=0.01)

    if hierarchy:
        base_kernel_class = kernel_class

        # select dimensions of X that specify hierarchy divisions
        hierarchy_dims = []
        if 'hzrd' in hierarchy:
            hierarchy_dims.append(n_lags + 1)
        if 'mouse' in hierarchy:
            hierarchy_dims.append(n_lags + 2)

        def kernel_class(*args, **kwargs):
            kernel_parent = base_kernel_class(*args, **kwargs)
            kernel_child = base_kernel_class(*args, **kwargs)
            return Hierarchical(
                [kernel_parent, kernel_child], indicator_dims=hierarchy_dims
            )

    # kernels not depending on inputs
    if kernel_type in ['White', 'Constant', 'Bias']:
        kernel = kernel_class(n_lags + 2)

    # kernel with all inputs together
    elif kernel_input == 'full':
        kernel = kernel_class(n_lags + 1, ARD=True)

    # kernel with only time
    elif kernel_input == 'time':
        kernel = kernel_class(1, active_dims=[n_lags])

    # kernel with only log-time
    elif kernel_input == 'logtime':
        base_kernel = kernel_class(1, active_dims=[n_lags])
        kernel = LogTranformedKernel(base_kernel)

    # kernel with warped time
    elif kernel_input == 'wtime':
        base_kernel = kernel_class(1, active_dims=[n_lags])
        kernel = WarpedKernel(base_kernel, n_tanh=n_tanh)

    # kernel with only hazard block input
    elif kernel_input == 'hzrd':
        kernel = kernel_class(1, active_dims=[n_lags + 1])

    # kernel with only stimulus, possibly linearly projected on a lower space
    elif kernel_input == 'stim':
        kernel = kernel_class(n_lags, ARD=True)

    elif kernel_input == 'proj':
        kernel_dims = np.arange(n_proj) + n_lags + 3
        base_kernel = kernel_class(n_proj, active_dims=kernel_dims)
        kernel = ProjKernel(base_kernel, n_lags, n_proj)
        if sigma is not None:
            kernel.W.prior = gpflow.priors.Laplace(0, sigma)

    elif kernel_input == 'expproj':
        kernel_dims = np.arange(n_proj) + n_lags + 3
        base_kernel = kernel_class(n_proj, active_dims=kernel_dims)
        kernel = ExpProjKernel(base_kernel, n_lags, n_proj)

    else:
        ValueError('Unknown kernel input type {}.'.format(kernel_input))

    return kernel


class PartialSVGP(gpflow.models.svgp.SVGP):
    """SVGP allowing partial predictions for additive kernels"""
    # "inspired" from https://gist.github.com/mrksr/6f020d6ce75ece9e0e1df16df21ef47a  # NOQA

    @gpflow.autoflow((gpflow.settings.float_type, [None, None]))
    def predict_f_partial(self, Xnew):
        return [
            self.build_partial_predict(Xnew, k) for k in self.kern.kernels
        ]

    @gpflow.params_as_tensors
    def build_partial_predict(self, Xnew, kern):
        """predict component contribution in an additive kernel"""

        float_type = gpflow.settings.float_type
        jitter = gpflow.settings.jitter

        # compute kernel stuff
        Z = self.feature.Z
        num_data = tf.shape(Z)[0]  # M
        Kmn = kern.K(Z, Xnew)
        Kmm = self.kern.K(Z) + tf.eye(num_data, dtype=float_type) * jitter
        Lm = tf.cholesky(Kmm)

        # compute the projection matrix A
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

        # another backsubstitution in the unwhitened case
        if not self.whiten:
            A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

        # construct the conditional mean
        fmean = tf.matmul(A, self.q_mu, transpose_a=True)

        return fmean


def build_model(dset, n_lags, max_nt, kernels_type, kernels_input, hierarchy,
                combination, n_z, batch_size, fast_init=False,
                mean_type='zero', hazard='nonsplit', **kernel_kwargs):
    """classification GP to fit reaction-time"""

    # prepare training data
    X_train, y_train = prepare_Xy(dset, n_lags, max_nt)
    X_train, y_train = np.vstack(X_train), np.vstack(y_train)

    # kernel for Gaussian process
    if len(kernels_type) == 1:
        kernels_type = list(kernels_type) * len(kernels_input)

    with gpflow.defer_build():
        kernels = [
            build_kernel(k_type, k_input, hierarchy, n_lags, **kernel_kwargs)
            for k_type, k_input in zip(kernels_type, kernels_input)
        ]

    kernel_op = operator.mul if combination == 'mul' else operator.add
    kernel = reduce(kernel_op, kernels)

    # inducing points
    hazard_codes = np.unique(X_train[:, -2])
    mice_codes = np.unique(X_train[:, -1])

    if fast_init:
        nz_total = n_z * len(mice_codes) * len(hazard_codes)
        Z = np.zeros((nz_total, X_train.shape[1]))

    else:
        Zs = []
        for i, j in product(hazard_codes, mice_codes):
            mask = np.all(X_train[:, -2:] == (i, j), axis=-1)
            kmeans = MiniBatchKMeans(n_z).fit(X_train[mask])
            Zs.append(kmeans.cluster_centers_.copy())
        Z = np.vstack(Zs)

    # sparse variational GP model
    likelihood = gpflow.likelihoods.Bernoulli(invlink=tf.nn.sigmoid)

    if mean_type == 'zero':
        mean_func = None
    elif mean_type == 'constant':
        y_mean = y_train.mean()
        base_rate = np.log(y_mean / (1 - y_mean))
        mean_func = gpflow.mean_functions.Constant(base_rate)
    elif mean_type == 'linear':
        y_mean = y_train.mean()
        base_rate = np.log(y_mean / (1 - y_mean))
        mean_func = gpflow.mean_functions.Linear(
            A=np.zeros((X_train.shape[1], 1)), b=base_rate
        )
    else:
        raise ValueError('Unknown mean function type {}.'.format(mean_type))

    model = PartialSVGP(
        X_train, y_train, kern=kernel, likelihood=likelihood, Z=Z,
        minibatch_size=batch_size, mean_function=mean_func
    )

    # attach parameters to the model
    model.n_lags = n_lags
    model.max_nt = max_nt

    return model


def build_ard_priors(model_kernel):
    """create ARD priors dictionary for projected kernel hyperparameters"""

    float_type = gpflow.settings.float_type
    gamma_prior = Gamma(float_type(0.001), float_type(0.001))

    priors = {}
    extra_factors = {}
    kernel_stack = [model_kernel]
    while kernel_stack:
        kernel = kernel_stack.pop()

        if isinstance(kernel, ProjKernel):
            # create an ARD-like prior, as in probabilistic PCA
            prec_name = kernel.W.pathname + '/precision'
            prec_shape = (1, kernel.W.shape[1])
            prec_factor = build_factor(prec_name, gamma_prior, prec_shape)
            extra_factors[prec_name] = prec_factor

            scale_sample = 1. / tf.sqrt(prec_factor.sample)
            priors[kernel.W] = Normal(float_type(0), scale_sample)

            kernel_stack.append(kernel.base_kernel)

        elif isinstance(kernel, gpflow.kernels.Combination):
            kernel_stack.extend(kernel.kernels)

    return priors, extra_factors


def build_model_ard(*args, **kwargs):
    """instantiate a GP model with ARD prior, using ADVI to fit posterior"""

    with gpflow.defer_build():
        model = build_model(*args, **kwargs)
    n_lags, max_nt = model.n_lags, model.max_nt

    priors, extra_factors = build_ard_priors(model.kern)
    model = FVGP(
        model.X.value, model.Y.value, model.kern, model.likelihood,
        Z=model.feature.Z.value, mean_function=model.mean_function,
        minibatch_size=model.X.batch_size,
        priors=priors, extra_factors=extra_factors
    )
    model.n_lags, model.max_nt = n_lags, max_nt

    return model


def predict_logpmf(model, dset, n_samples):
    """approximate posterior log-PMF of the discrete time-survival model"""

    # transform data for the model
    Xs = prepare_X(dset, model.n_lags, model.max_nt)

    log_pmf_trials = []
    for i, X in enumerate(Xs):
        # sample discrete-time hazard functions
        samples = model.predict_f_samples(X, n_samples).squeeze()
        hazard = expit(samples)  # inv-logit transform

        # convert into probability mass functions
        log_sf = np.log(1 - hazard).cumsum(axis=1)
        log_sf = np.hstack([np.zeros((n_samples, 1)), log_sf[:, :-1]])
        log_pmf = log_sf + np.log(hazard)

        # average to get Monte-Carlo estimate of the posterior PMF
        log_pmf = logsumexp(log_pmf, axis=0) - np.log(n_samples)

        # append no-lick probability at the end
        log_prob_lick = logsumexp(log_pmf)
        if np.isclose(log_prob_lick, 0):
            log_pmf -= log_prob_lick  # ajust probabilities to sum to 1
            log_prob_nolick = -np.inf  # P(no-lick) = 0
        else:
            log_prob_nolick = np.log1p(-np.exp(log_prob_lick))
        log_pmf = np.append(log_pmf, log_prob_nolick)

        assert np.all(log_pmf <= 0)  # detect numerical errors

        log_pmf_trials.append(log_pmf)

    return log_pmf_trials
