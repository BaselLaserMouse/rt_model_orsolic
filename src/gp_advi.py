from itertools import chain
from collections import namedtuple

import numpy as np
import scipy.special as sp
import gpflow
from gpflow.params import Parameter
import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Gamma, bijectors


def get_support_transform(distribution):
    """get transform from unconstrained support to distribution support"""
    if isinstance(distribution, Gamma):
        return bijectors.Exp()
    return bijectors.Identity()


Factor = namedtuple(
    'Factor', ['sample', 'elbo_part', 'tensors', 'init_tensors']
)


def build_factor(name, prior, shape=None):
    """instantiate an approximate posterior factor"""

    # create posterior Gaussian factor
    shape = shape or prior.batch_shape
    float_type = gpflow.settings.float_type
    with tf.variable_scope(name):
        init_loc = tf.placeholder(float_type, shape=shape)
        init_log_scale = tf.placeholder(float_type, shape=shape)
        loc = tf.get_variable('loc', initializer=init_loc)
        log_scale = tf.get_variable('log_scale', initializer=init_log_scale)
        scale = tf.exp(log_scale, name='scale')

    # contribution to the ELBO
    transform = get_support_transform(prior)

    raw_sample = Normal(loc, scale).sample()
    sample = transform.forward(raw_sample)
    log_abs_det_jac = transform.forward_log_det_jacobian(
        raw_sample, transform.forward_min_event_ndims
    )
    prior_logprob = prior.log_prob(sample)
    entropy = 0.5 * (1.0 + np.log(2 * np.pi)) + log_scale
    elbo_part = tf.reduce_sum(
        prior_logprob + log_abs_det_jac + entropy
    )

    tensors = (loc, log_scale)
    init_tensors = (init_loc, init_log_scale)
    return Factor(sample, elbo_part, tensors, init_tensors)


class FVGP(gpflow.models.SVGP):

    def __init__(self, *args, priors=None, extra_factors=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._advi_values = {}
        self.factors = {}

        # use additional posterior factors, create corresponding initial values
        if extra_factors:
            self.factors.update(extra_factors)
            for name, factor in extra_factors.items():
                factor_shape = factor.sample.shape
                self._advi_values[name] = (
                    np.zeros(factor_shape), np.full(factor_shape, -5)
                )

        # transform hyperparameters to use approximate posterior samples
        for parameter, prior in priors.items():
            factor = build_factor(parameter.pathname, prior, parameter.shape)
            self.factors[parameter.pathname] = factor
            self._advi_values[parameter.pathname] = (
                np.zeros(parameter.shape), np.full(parameter.shape, -5)
            )

            parent = parameter.parent
            name = next(key for key, value in parent.children.items()
                        if value is parameter)
            parent.unset_child(name, parameter)

            new_parameter = Parameter(factor.sample, trainable=False)
            parent.set_child(name, new_parameter)

        self._advi_initializables = [
            (tensor, tf.is_variable_initialized(tensor))
            for tensor in self.advi_tensors
        ]

    @property
    def advi_tensors(self):
        tensors = (factor.tensors for factor in self.factors.values())
        return list(chain.from_iterable(tensors))

    @property
    def initializables(self):
        return super().initializables + self._advi_initializables

    @property
    def initializable_feeds(self):
        feeds = super().initializable_feeds
        for name, factor in self.factors.items():
            init_loc, init_log_scale = factor.init_tensors
            loc, log_scale = self._advi_values[name]
            feeds[init_loc], feeds[init_log_scale] = loc, log_scale
        return feeds

    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.advi_tensors

    def anchor(self, session):
        super().anchor(session)
        for name, factor in self.factors.items():
            self._advi_values[name] = session.run(factor.tensors)

    def read_values(self, session=None):
        values = {}
        for parameter in self.parameters:
            if parameter.pathname in self.factors:
                continue
            values[parameter.pathname] = parameter.read_value(session)

        if session is None:
            for name, factor in self.factors.items():
                values[name] = self._advi_values[name]
        else:
            for name, factor in self.factors.items():
                values[name] = session.run(factor.tensors)

        return values

    def assign(self, values, session=None, force=True):
        values = values.copy()
        for name in self._advi_values:
            if name in values:
                self._advi_values[name] = values.pop(name)
        super().assign(values, session, force)
        self.initialize(session, force)

    def _build_objective(self, likelihood, prior):
        objective = super()._build_objective(likelihood, prior)
        elbo_parts = [factor.elbo_part for factor in self.factors.values()]
        return objective - tf.reduce_sum(elbo_parts)

    def predict_density(self, Xnew, Ynew, nsamples=50):
        samples = [
            super(FVGP, self).predict_density(Xnew, Ynew)
            for _ in range(nsamples)
        ]
        return sp.logsumexp(samples, 0) - np.log(nsamples)

    def predict_y(self, Xnew):
        raise NotImplementedError()

    def predict_f(self, Xnew):
        raise NotImplementedError()

    def predict_f_full_cov(self, Xnew):
        raise NotImplementedError()

    def predict_f_samples(self, Xnew, num_samples):
        return np.vstack([
            super(FVGP, self).predict_f_samples(Xnew, 1)
            for _ in range(num_samples)
        ])
