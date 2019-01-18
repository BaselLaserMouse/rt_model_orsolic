#!/usr/bin/env python3

from pathlib import Path
from itertools import product

import defopt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sb

from strenum import strenum


def sample_rt(dset, n_samples):
    """sample reaction-time for each trial using predicted log-PMF"""
    n_trials = len(dset)

    rt = np.empty((n_trials, n_samples))
    for i, (_, row) in enumerate(dset.iterrows()):
        pmf = np.exp(row.log_pmf)
        rt[i, :] = np.argmax(np.random.multinomial(1, pmf, n_samples), -1)
        rt[i, rt[i, :] == len(pmf) - 1] = np.nan

    return rt


def load_data(pred_filename, n_samples, folds):
    """load predictions dataset and draw predictive samples"""

    dset = pd.read_pickle(pred_filename)

    mask = False
    for fold in folds:
        mask |= dset[fold]
    dset = dset[mask].copy()

    # convert to base 2
    # TODO move to matlab code
    dset['sig'] /= np.log(2)

    # sample predicted reaction-time on test dataset
    samples = sample_rt(dset, n_samples)

    dset_base = dset[['sig', 'mouse', 'hazard', 'change']]
    dset_pred = pd.concat([
        pd.DataFrame({'rt': sample, 'sample_id': i, **dset_base})
        for i, sample in enumerate(samples.T)
    ])

    dset_pred['rt_change'] = dset_pred['rt'] - dset_pred['change']

    # add early/correct/late columns to datasets
    dset['early'] = dset['rt'] <= dset['change']
    dset['hit'] = dset['rt'] > dset['change']
    dset['miss'] = np.isnan(dset['rt'])

    dset_pred['early'] = dset_pred['rt'] <= dset_pred['change']
    dset_pred['hit'] = dset_pred['rt'] > dset_pred['change']
    dset_pred['miss'] = np.isnan(dset_pred['rt'])

    return dset, dset_pred


def plot_licks(dset_test, dset_pred, lick_col, rt_col, axes, titles=True,
               xlabels=True):

    # lick proportion distribution
    lick_pred = dset_pred.groupby('sample_id').agg({lick_col: np.mean})

    sb.distplot(lick_pred, ax=axes[0])
    axes[0].axvline(dset_test[lick_col].mean(), color='k', lw=2)

    # plot distribution for each predictive sample
    for i in dset_pred.sample_id.unique():
        df_pred = dset_pred[(dset_pred.sample_id == i) & dset_pred[lick_col]]
        rt_pred = df_pred[rt_col]

        sb.kdeplot(rt_pred, cut=0, alpha=0.1, color='b', legend=False,
                   ax=axes[1])
        sb.kdeplot(rt_pred, cut=0, alpha=0.1, color='b', legend=False,
                   ax=axes[2], cumulative=True)

    # plot dataset distribution
    df_test = dset_test[dset_test[lick_col]]
    rt_test = df_test[rt_col]

    sb.kdeplot(rt_test, cut=0, color='k', lw=2, legend=False, ax=axes[1])
    sb.kdeplot(rt_test, cut=0, color='k', lw=2, legend=False, ax=axes[2],
               cumulative=True)

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    if xlabels:
        axes[0].set_xlabel('proportion')
        axes[1].set_xlabel('time (frame triplet)')
        axes[2].set_xlabel('time (frame triplet)')

    if titles:
        axes[0].set_title('{} licks proportion distribution'.format(lick_col))
        axes[1].set_title('{} licks time distribution'.format(lick_col))
        axes[2].set_title('{} licks time cumulative distribution'
                          .format(lick_col))


def plot_early_licks(dset_test, dset_pred, fig_title=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_licks(dset_test, dset_pred, 'early', 'rt', axes)
    if fig_title:
        fig.suptitle(fig_title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()
    return fig


def plot_hit_licks(dset_test, dset_pred, fig_title=None):
    sigs = sorted(dset_test.sig.unique())
    n_sigs = len(sigs)

    fig, axes = plt.subplots(
        n_sigs, 3, sharex='col', sharey='col', figsize=(10, 12), squeeze=False
    )

    for i, (ax, sig) in enumerate(zip(axes, sigs)):
        df_test = dset_test[dset_test.sig == sig]
        df_pred = dset_pred[dset_pred.sig == sig]
        plot_licks(df_test, df_pred, 'hit', 'rt_change', ax,
                   i == 0, i == n_sigs-1)

    for ax, sig in zip(axes[:, 0], sigs):
        ax.set_ylabel('stimulus {:.2f}'.format(sig), size='large')

    if fig_title:
        fig.suptitle(fig_title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()
    return fig


def plot_psycho_chrono(dset_test, dset_pred, fig_title=None):
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(12, 4))

    # early licks proportions
    earlylicks_pred = (
        dset_pred.groupby(['sig', 'sample_id']).agg({'early': 'mean'})
    )
    earlylicks_test = dset_test.groupby('sig').agg({'early': 'mean'})

    axes[0].plot(earlylicks_pred.unstack(), 'b', alpha=0.1)
    axes[0].plot(earlylicks_test, '-ok', lw=2)
    axes[0].grid()
    axes[0].set_xlabel('change magnitude (octaves)')
    axes[0].set_ylabel('proportion')
    axes[0].set_title('Early licks proportion')
    axes[0].set_ylim([0, 1])

    # psychometric curve, i.e. hit / (hit + miss)
    hitlicks_pred = (
        dset_pred[~dset_pred['early']]
        .groupby(['sig', 'sample_id']).agg({'hit': 'mean'})
    )
    hitlicks_test = (
        dset_test[~dset_test['early']]
        .groupby('sig').agg({'hit': 'mean'})
    )

    axes[1].plot(hitlicks_pred.unstack(), 'b', alpha=0.1)
    axes[1].plot(hitlicks_test, '-ok', lw=2)
    axes[1].grid()
    axes[1].set_xlabel('change magnitude (octaves)')
    axes[1].set_ylabel('proportion')
    axes[1].set_title('Psychometric function')
    axes[1].set_ylim([0, 1])

    # chronometric curve
    period = 0.05
    hitrt_pred = period * (
        dset_pred[dset_pred['hit'] & (dset_pred['sig'] > 0)]
        .groupby(['sig', 'sample_id']).agg({'rt_change': 'mean'})
    )
    hitrt_test = period * (
        dset_test[dset_test['hit'] & (dset_test['sig'] > 0)]
        .groupby('sig').agg({'rt_change': 'mean'})
    )

    axes[2].plot(hitrt_pred.unstack(), 'b', alpha=0.1)
    axes[2].plot(hitrt_test, '-ok', lw=2)
    axes[2].grid()
    axes[2].set_xlabel('change magnitude (octaves)')
    axes[2].set_ylabel('reaction time (s)')
    axes[2].set_title('Chronometric function')

    if fig_title:
        fig.suptitle(fig_title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()
    return fig


Fold = strenum('Fold', 'train val test')


def main(pred_filename, figure_dir, *, folds=('test',), n_samples=100):
    """Sample the predictive posterior distribution and plot results

    :param str pred_filename: Pandas dataset file (.pickle format) with
                              predicted log-PMF
    :param str figure_dir: directory for generated figures
    :param list[Fold] folds: data folds to use
    :param int n_samples: number of samples

    """

    # create output directory
    figure_path = Path(figure_dir)
    figure_path.mkdir(parents=True, exist_ok=True)

    # load dataset (and draw predictive samples)
    dset, dset_pred = load_data(pred_filename, n_samples, folds)

    # early and hit licks distribution, psycho/chonometric curves
    with PdfPages(str(figure_path / 'early_licks.pdf')) as pdf_early, \
            PdfPages(str(figure_path / 'hit_licks.pdf')) as pdf_hit, \
            PdfPages(str(figure_path / 'early_hit_chrono.pdf')) as pdf_chrono:

        mice = dset.mouse.unique()
        hazards = dset.hazard.unique()

        if len(mice) > 1 or len(hazards) > 1:
            fig_early = plot_early_licks(dset, dset_pred, "Population")
            pdf_early.savefig(fig_early)
            plt.close(fig_early)

            fig_hit = plot_hit_licks(dset, dset_pred, "Population")
            pdf_hit.savefig(fig_hit)
            plt.close(fig_hit)

            fig_chrono = plot_psycho_chrono(dset, dset_pred, "Population")
            pdf_chrono.savefig(fig_chrono)
            plt.close(fig_chrono)

        for mouse, hazard in product(mice, hazards):
            fig_title = '{} / {}'.format(mouse, hazard)

            mouse_test = dset[dset.mouse == mouse]
            mouse_test = mouse_test[mouse_test.hazard == hazard]
            mouse_pred = dset_pred[dset_pred.mouse == mouse]
            mouse_pred = mouse_pred[mouse_pred.hazard == hazard]

            fig_early = plot_early_licks(mouse_test, mouse_pred, fig_title)
            pdf_early.savefig(fig_early)
            plt.close(fig_early)

            fig_hit = plot_hit_licks(mouse_test, mouse_pred, fig_title)
            pdf_hit.savefig(fig_hit)
            plt.close(fig_hit)

            fig_chrono = plot_psycho_chrono(mouse_test, mouse_pred, fig_title)
            pdf_chrono.savefig(fig_chrono)
            plt.close(fig_chrono)


if __name__ == "__main__":
    defopt.run(main)
