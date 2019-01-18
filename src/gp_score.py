#!/usr/bin/env python3

import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import defopt


def compress_method_names(methods):
    """remove starting and ending common sub-strings in methods names"""

    # remove everything if only one element
    if len(methods) == 1:
        return [""]

    idx_start, _ = next(it.dropwhile(
        lambda x: len(set(x[1])) == 1, enumerate(zip(*methods))
    ))

    reversed_methods = [reversed(m) for m in methods]
    common_end, _ = next(it.dropwhile(
        lambda x: len(set(x[1])) == 1, enumerate(zip(*reversed_methods))
    ))
    idx_end = -common_end if common_end > 0 else None

    return [method[idx_start:idx_end] for method in methods]


def main(figure_dir, *pred_filename, labels=None, for_paper=False):
    """Score GP model fit and plot results

    :param str figure_dir: directory for generated figures
    :param str pred_filename: Pandas dataset file (.pickle format) with
                              predicted log-PMF
    :param list[str] labels: method name for each input dataset
    :param bool for_paper: use settings for the paper panel

    """

    if labels and len(pred_filename) != len(labels):
        raise ValueError(
            "Inconsistent number of dataset filenames ({}) and labels ({})"
            .format(len(pred_filename), len(labels))
        )

    # define output directory
    figure_path = Path(figure_dir)
    figure_path.mkdir(parents=True, exist_ok=True)

    # load datasets
    dsets = (pd.read_pickle(fname) for fname in pred_filename)

    cols = ['sig', 'mouse', 'log_pmf', 'rt', 'train', 'test', 'val']
    methods = labels if labels else compress_method_names(pred_filename)

    dset = pd.concat([
        pd.DataFrame(data={'method': method, **dset[cols]})
        for method, dset in zip(methods, dsets)
    ])

    # convert to base 2
    # TODO move to matlab code
    dset['sig'] /= np.log(2)

    # extract predictive log-likelihood at observation datapoint
    dset['logprob'] = dset.apply(
        lambda x: x.log_pmf[-1 if np.isnan(x.rt) else int(x.rt)], axis=1
    )

    dset['split'] = 'train'
    dset.loc[dset.val, 'split'] = 'valid.'
    dset.loc[dset.test, 'split'] = 'test'

    # save numerical results
    dset = dset.drop('log_pmf', axis=1)
    dset.to_csv(figure_path / 'logprob.csv')

    # plot summary results
    hue_order = ['train', 'valid.', 'test']
    hue_order = [hue for hue in hue_order if any(dset['split'] == hue)]

    if for_paper:
        # seaborn settings
        rc_params = {
            'font.sans-serif': ['Arial'],
            'font.size': 8,
            'lines.linewidth': 0.5,
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'axes.titlesize':8,
            'axes.labelsize':8,
            'xtick.major.size':1,
            'ytick.major.size':1
        }
        sb.set(style='ticks')
        sb.set_context('paper', rc=rc_params)    
        # text as type rather than outlines
        plt.rcParams['pdf.fonttype'] = 'truetype'

        fig, ax = plt.subplots(figsize=(2, 2))
    else:
        sb.set(style='ticks')
        fig, ax = plt.subplots()

    sb.pointplot(
        x='method', y='logprob', data=dset, hue='split', hue_order=hue_order,
        join=False, dodge=0.3, ax=ax
    )

    if for_paper:
        ax.set_ylim(-4.7, -3.6)
        ax.get_legend().remove()

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    sb.despine(fig, offset=3, trim=True)
    fig.tight_layout()
    fig.savefig(str(figure_path / 'logprob_mean.pdf'))

    g = sb.factorplot(
        data=dset, x='method', y='logprob', hue='split', hue_order=hue_order,
        col='sig', col_wrap=3, join=False, dodge=0.3
    )
    g.set_xticklabels(rotation=30)
    for ax in g.axes.ravel():
        ax.grid()
    
    g.fig.tight_layout()
    g.fig.savefig(str(figure_path / 'logprob_sig.pdf'))


if __name__ == "__main__":
    defopt.run(main)
