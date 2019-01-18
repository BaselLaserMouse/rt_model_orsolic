#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import defopt


def plot_gamma_posterior(loc, scale, n_samples):
    """plot the posterior of the precision of the projection components"""

    n_comps = len(loc)
    samples = np.exp(np.random.randn(n_samples, n_comps) * scale + loc)

    fig, axes = plt.subplots(n_comps, 2, sharex='col', figsize=(10, 12))
    for ax, component_samples in zip(axes, samples.T):
        ax[0].grid()
        sb.distplot(component_samples, ax=ax[0])
        ax[1].grid()
        sb.distplot(1 / np.sqrt(component_samples), ax=ax[1])

    axes[0, 0].set_title('Precision posteriors')
    axes[0, 1].set_title('Standard deviation posteriors')

    fig.tight_layout()
    return fig


def plot_W_posterior(loc, scale):
    """plot the posterior distribution of projection matrix"""

    n_lags, n_comps = loc.shape
    xs = np.arange(n_lags)

    n_rows = int(np.ceil(n_comps / 3))
    fig, axes = plt.subplots(
        n_rows, 3, sharex=True, sharey=True, figsize=(10, 15)
    )
    for ax, mu, sigma in zip(axes.flat, loc.T, scale.T):
        ax.grid()
        ax.fill_between(xs, mu - 2 * sigma, mu + 2 * sigma, color='b',
                        alpha=0.2, lw=0, label='mean + 2 sd')
        ax.fill_between(xs, mu - sigma, mu + sigma, color='b', alpha=0.2, lw=0,
                        label='mean + 1 sd')
        ax.plot(mu, c='k', lw=1.5, label='mean')

    axes[-1, -1].legend()
    fig.tight_layout()

    return fig


def plot_positive_posterior(loc, scale, n_samples):
    """plot the posterior for a strictly positive kernel hyperparameter"""

    samples = np.exp(np.random.randn(n_samples) * scale + loc)

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.grid()
    sb.distplot(samples, ax=ax)

    fig.tight_layout()
    return fig


def main(result_dir, figure_dir, *, n_samples=1000):
    """Plot approximate posterior distributions for GP model hyperparameters

    :param str result_dir: directory for results files
    :param str figure_dir: directory for generated figures
    :param int n_samples: number of samples from posterior

    """

    # define input and output directories
    result_path = Path(result_dir)
    figure_path = Path(figure_dir)
    figure_path.mkdir(parents=True, exist_ok=True)

    # load model parameters
    model_params = np.load(result_path / 'model_params_best.npz')

    # plot posterior distributions
    keys = [k for k in sorted(model_params) if k.startswith('FVGP/kern')]
    for key in keys:
        params = model_params[key]

        if not key.endswith('W'):
            continue

        loc, scale = params[0], np.exp(params[1])
        figname = key.replace('/', '_')

        # retrieve corresponding precision parameter
        gamma_key = key.replace('FVGP', 'PartialSVGP') + '/precision'
        gamma_loc = model_params[gamma_key][0, 0, :]
        gamma_scale = np.exp(model_params[gamma_key][1, 0, :])

        # sort by increasing (log)-precision
        idx = np.argsort(gamma_loc)
        gamma_loc, gamma_scale = gamma_loc[idx], gamma_scale[idx]
        loc, scale = loc[:, idx], scale[:, idx]

        # flip to make bigger deviation positive
        flip_mask = loc.mean(0) < 0
        loc[:, flip_mask] = -loc[:, flip_mask]

        # plot posterior distributions
        gamma_figname = key.replace('/', '_') + '_precision.png'
        fig = plot_gamma_posterior(gamma_loc, gamma_scale, n_samples)
        fig.savefig(str(figure_path / gamma_figname))

        fig = plot_W_posterior(loc, scale)

        fig.savefig(str(figure_path / figname))


if __name__ == "__main__":
    defopt.run(main)
