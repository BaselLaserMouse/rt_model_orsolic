#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import gpflow
from tensorflow.contrib.distributions import Gamma
import matplotlib.pyplot as plt
import seaborn as sb

from gp_advi import FVGP


def generate_dataset(n_points):
    """create a synthetic dataset"""
    X = np.sort(np.random.rand(n_points, 1) * np.pi * 20, axis=0)
    y = np.sin(X / 10) * 3 + np.random.randn(n_points, 1)
    return X, y


def main():
    np.random.seed(1234)
    X, y = generate_dataset(200)

    float_type = gpflow.settings.float_type
    prior = Gamma(float_type(0.01), float_type(0.01))

    with gpflow.defer_build():
        kern = gpflow.kernels.RBF(1)
    likelihood = gpflow.likelihoods.Gaussian()
    priors = {kern.variance: prior, kern.lengthscales: prior}
    Z = np.linspace(min(X), max(X), 10)[:, np.newaxis]
    model = FVGP(X, y, kern, likelihood, Z=Z, priors=priors)

    result_path = Path('advi_model_params.npz')
    if result_path.exists():
        print('loading parameters values from {}'.format(result_path))
        params = dict(np.load(str(result_path)))
        model.assign(params)
    else:
        optimizer = gpflow.training.AdamOptimizer(0.01)
        optimizer.minimize(model, maxiter=5000)
        np.savez(str(result_path), **model.read_values())

    sess = model.enquire_session()
    var_samples = [kern.variance.read_value(sess) for _ in range(2000)]
    len_samples = [kern.lengthscales.read_value(sess) for _ in range(2000)]

    xs = np.linspace(X[0, 0], X[-1, 0], 500)[:, np.newaxis]
    f_samples = [model.predict_f_samples(xs, 1) for _ in range(200)]
    f_samples = np.vstack(f_samples)[..., 0]

    # TODO fix model printing?
    # print(model)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].grid()
    axes[0].plot(X, y, 'o')
    axes[0].plot(xs, f_samples.T, 'k', alpha=0.1)
    axes[0].plot(xs, np.mean(f_samples, 0), lw=2)
    axes[1].set_title('kernel variance posterior')
    axes[1].grid()
    sb.distplot(var_samples, ax=axes[1])
    axes[2].set_title('kernel lengthscale posterior')
    axes[2].grid()
    sb.distplot(len_samples, ax=axes[2])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
