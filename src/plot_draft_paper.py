#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
import defopt

from gp_ppc import load_data


def plot_psycho(dset_pred, ax, label):
    """plot predictive psychometric curve"""

    hitlicks_pred = (
        dset_pred[~dset_pred['early']]
        .groupby(['sig', 'sample_id']).agg({'hit': 'mean'})
        .unstack()
    )
    hitlicks_mean = hitlicks_pred.mean(axis=1)
    hitlicks_prc = hitlicks_pred.quantile([0.025, 0.975], axis=1)

    ax.fill_between(
        hitlicks_pred.index, hitlicks_prc.loc[0.025], hitlicks_prc.loc[0.975],
        alpha=0.3, lw=0, color='k'
    )

    ax.plot(hitlicks_mean, color='k', label=label, alpha=0.8)


def plot_chrono(dset_pred, period, ax):
    """plot predictive chronometric curve"""

    hitrt_pred = period * (
        dset_pred[dset_pred['hit'] & (dset_pred['sig'] > 0)]
        .groupby(['sig', 'sample_id']).agg({'rt_change': 'median'})
        .unstack()
    )
    hitrt_mean = hitrt_pred.median(axis=1)
    hitrt_prc = hitrt_pred.quantile([0.025, 0.975], axis=1)

    ax.fill_between(
        hitrt_pred.index, hitrt_prc.loc[0.025], hitrt_prc.loc[0.975],
        alpha=0.3, lw=0, color='k'
    )

    ax.plot(hitrt_mean, color='k', alpha=0.8)


def plot_rt_cdf(dset_pred, rt_range, period, color, ax):
    """plot cumulative density function for reaction time"""
    #dset_pred = dset_pred[dset_pred.rt > 20]
    sample_groups = dset_pred[~dset_pred['miss']].groupby('sample_id')
    cdf_pred = np.zeros((len(sample_groups), len(rt_range)))
    for i, (_, group) in enumerate(sample_groups):
        cdf_pred[i] = np.mean(
            group.rt.values[:, np.newaxis] <= rt_range, axis=0
        )
    cdf_perc = np.percentile(cdf_pred, [2.5, 97.5], axis=0)
    cdf_mean = cdf_pred.mean(0)

    rt_test_sec = rt_range * period
    ax.fill_between(rt_test_sec, cdf_perc[0], cdf_perc[1], alpha=0.3, lw=0,
                    color=color, edgecolor=color)
    ax.plot(rt_test_sec, cdf_mean, alpha=0.8, color=color)


def plot_early_lick_hazard(dset_pred, rt_bins, period, color, ax):
    sample_groups = dset_pred.groupby('sample_id')
    hazard_pred = np.zeros((len(sample_groups), len(rt_bins)-1))
    for i, (_, group) in enumerate(sample_groups):
        hazard_pred[i] = get_early_lick_hazard(group, rt_bins)

    hazard_perc = np.percentile(hazard_pred, [2.5, 97.5], axis=0)
    hazard_mean = hazard_pred.mean(0)

    rt_bins_sec = rt_bins[1:] * period
    ax.fill_between(rt_bins_sec, hazard_perc[0], hazard_perc[1], alpha=0.3,
                    lw=0, color=color, edgecolor=color)
    ax.plot(rt_bins_sec, hazard_mean, alpha=0.8, color=color)


def get_early_lick_hazard(dset, rt_bins):
    rt = np.sort(dset[dset['early']]['rt'].values)

    licks_test, _ = np.histogram(rt, bins=rt_bins)
    changes_hit, _ = np.histogram(dset[dset['hit']]['change'].values, bins=rt_bins)
    changes_miss, _ = np.histogram(dset[dset['miss']]['change'].values, bins=rt_bins)

    completed_trials = np.append(0, np.cumsum(changes_hit[:-1])) + \
        np.append(0, np.cumsum(changes_miss[:-1])) + \
        np.append(0, np.cumsum(licks_test[:-1]))
    total_trials = dset.shape[0]
    licks_hazard = licks_test / (total_trials - completed_trials)
    return licks_hazard


def plot_psycho_chrono(dset_test, dset_gp, filters, axes, early_licks):
    """display chronometric and psychometric curves side by side"""

    # psychometric curve, i.e. hit / (hit + miss)
    hitlicks_test = (
        dset_test[~dset_test['early']]
        .groupby('sig').agg({'hit': 'mean'})
    )
    hplot = axes[0].plot(hitlicks_test, '--.', dashes=(4, 4),
                         color=[1.0, 0.1, 0.1], label='Holdout data', ms=4)
    for hline in hplot:
        hline.set_clip_on(False)

    plot_psycho(dset_gp, axes[0], label='Model prediction')

    axes[0].set_xlabel('Change magnitude (octaves)')
    axes[0].set_ylabel('Proportion hits')
    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(0, 2)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # chronometric curve
    period = 0.05
    hitrt_test = period * (
        dset_test[dset_test['hit'] & (dset_test['sig'] > 0)]
        .groupby('sig').agg({'rt_change': 'median'})
    )

    hplot = axes[1].plot(hitrt_test, '--.', dashes=(4,4),
                         color=[1.0, 0.1, 0.1], ms=4)
    for hline in hplot:
        hline.set_clip_on(False)
    plot_chrono(dset_gp, period, axes[1])

    axes[1].set_xlabel('Change magnitude (octaves)')
    axes[1].set_ylabel('Reaction time (s)')
    axes[1].axis('tight')
    axes[1].set_xlim(0, 2)
    if all(hitrt_test['rt_change'] > 0.5):
        axes[1].set_ylim(0.5, 1.5)
    else:
        axes[1].set_ylim(0, 1.5)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # Ivana's colormap for early / late blocks
    cmap = np.array([
         [1.0, 0.4, 0],
         [0.4, 0.2, 0.8]
    ])

    # cumulative density function of early licks
    if early_licks:
        dset_test = dset_test[dset_test['early']]
        dset_gp = dset_gp[dset_gp['early']]

    #dset_test = dset_test[dset_test.rt > 20]
    for i, (hazard, dset_group) in enumerate(dset_test.groupby('hazard')):
        rt_range = np.linspace(0, 16, num=161) / period
        rt_test = np.sort(dset_group[~dset_group['miss']]['rt'].values)
        cdf_test = np.mean(rt_test[:, np.newaxis] <= rt_range, axis=0)

        axes[2].plot(rt_range * period, cdf_test, '--',
                     dashes=(4, 4), color=cmap[i])  # TODO use markers?
        plot_rt_cdf(dset_gp[dset_gp.hazard == hazard], rt_range, period,
                    color=cmap[i], ax=axes[2])

    axes[2].set_xlabel('Time from stimulus onset (s)')
    axes[2].set_ylabel('Cumul. lick proportion')
    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(0, 16)
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(5))

    # GP model filters
    xs = np.arange(len(filters)) * period
    axes[3].axhline(0, color='0.6', linestyle='--')
    axes[3].plot(xs, filters[:, 0], sb.xkcd_rgb['dark mauve'])
    axes[3].plot(xs, filters[:, 1], sb.xkcd_rgb['faded green'])
    axes[3].set_ylabel('Weight')
    axes[3].set_xlabel('Time lag (s)')
    axes[3].set_xlim(0, 2.5)
    #axes[3].set_ylim(-0.25, 0.5)
    #axes[3].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    axes[3].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #axes[3].set_yticks([-.25, 0, .5])
    #return fig, axes


def load_filters(model_path):
    """load and sort filters from a projected model"""

    params = np.load(str(model_path / 'model_params_best.npz'))
    filters = (value for param, value in params.items() if param.endswith('W'))
    filters = next(filters, None)

    # sort by filter standard deviation
    filters = filters[:, np.argsort(-filters.std(axis=0))]

    # flip to make bigger deviation positive
    mask_idx = np.arange(filters.shape[1])
    flip_mask = filters[np.abs(filters[0:10,:]).argmax(0), mask_idx] < 0
    filters[:, flip_mask] = -filters[:, flip_mask]

    return filters


def main(fname, *, supplement=False, early_licks=False, all_splits=False):
    """Plot model fit summaries

    :param str fname: output file name
    :param bool supplement: whether to print supplemental figure
    :param bool early_licks: whether to only plot timing of early licks
    :param bool all_splits: whether to use all splits
    """

    # set seaborn style, fix sans-serif font to avoid missing minus sign in pdf
    rc_params = {
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.major.size': 1,
        'ytick.major.size': 1
    }
    sb.set(style='ticks')
    sb.set_context('paper', rc=rc_params)
    plt.rcParams['pdf.fonttype'] = 'truetype'

    if supplement:
        models = [
            'IO_075__constant__matern52__proj_wtime__ard',
            'IO_078__constant__matern52__proj_wtime__ard',
            'IO_079__constant__matern52__proj_wtime__ard',
            'IO_081__constant__matern52__proj_wtime__ard',
            'IO_083__constant__matern52__proj_wtime__ard'
        ]
    else:
        models = [
            'IO_080__constant__matern52__proj_wtime__ard',
            'IO_080__constant__matern52__proj__ard',
            'IO_080__constant__matern52__wtime'
        ]

    fig, axes = plt.subplots(
        len(models), 4, figsize=(20/2.54, 4.25/2.54 * len(models))
    )

    for idx, model in enumerate(models):
        # define input models
        gp_path = Path('results_new', model)

        # fix random seed for reproducibitity
        np.random.seed(1234)

        # load model, make predictions and get filters
        # dset, _ = load_data(
        #     gp_path / 'predictions.pickle', 1, ('test', 'train', 'val')
        # )

        # print(model, 
        #     'train', sum(dset['train']), sum(dset[dset.hazard != 'nonsplit']['train']),
        #     'val', sum(dset['val']), sum(dset[dset.hazard != 'nonsplit']['val']),
        #     'test', sum(dset['test']), sum(dset[dset.hazard != 'nonsplit']['test']))

        # load model, make predictions and get filters
        if all_splits:
            dset_test, dset_gp = load_data(
                gp_path / 'predictions.pickle', 500, ('test', 'train', 'val')
            )
        else:
            dset_test, dset_gp = load_data(
                gp_path / 'predictions.pickle', 500, ('test',)
            )

        dset_test = dset_test[dset_test.hazard != 'nonsplit'].copy()
        dset_gp = dset_gp[dset_gp.hazard != 'nonsplit'].copy()

        model_opts = np.load(gp_path / 'model' / 'model_options.npz')
        if 'proj' in model_opts['kernels_input']:
            gp_filters = load_filters(gp_path / 'model')
        else:
            gp_filters = np.full((2, 2), np.nan)

        # create the figure and save it
        if axes.ndim > 1:
            plot_psycho_chrono(
                dset_test, dset_gp, gp_filters, axes[idx, :], early_licks
            )
        else:
            plot_psycho_chrono(dset_test, dset_gp, gp_filters, axes, early_licks)

    sb.despine(fig, offset=3, trim=False)
    fig.tight_layout()
    fig.savefig(fname)


if __name__ == "__main__":
    defopt.run(main)
