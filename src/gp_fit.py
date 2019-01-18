#!/usr/bin/env python3

import json
import time
from pathlib import Path
from functools import partial

import defopt
import numpy as np
import scipy.io as io
import pandas as pd
import tensorflow as tf
import gpflow
from sklearn.model_selection import train_test_split

from strenum import strenum
from gp_model import prepare_Xy, build_model, build_model_ard


def load_data(filename):
    """load reaction time dataset"""

    # convert matlab data into dataframe
    mat = io.loadmat(filename)
    dset = pd.DataFrame({
        'rt': mat['rt'].ravel() - 1,
        'sig': mat['sig'].ravel(),
        'sig_avg': mat['sig_avg'].ravel(),
        'sig_std': mat['sig_std'].ravel(),
        'session': mat['session'].ravel(),
        'hazard': np.vstack(mat['hazard'].ravel()).ravel(),
        'outcome': mat['outcome'].ravel(),
        'noiseless': mat['noiseless'].ravel() != 0,
        'ys': [y.ravel() for y in mat['ys'].flat],
        'change': mat['change'].ravel() - 1
    })

    # add reaction-time from change point
    # TODO move to matlab code
    dset['rt_change'] = dset.rt - dset.change

    # add filename as mouse name
    # TODO move to matlab code
    dset['mouse'] = Path(filename).stem

    # misc. cleaning
    # TODO move to matlab code
    dset = dset.groupby('sig').filter(lambda x: len(x) > 200)
    dset = dset[~dset.noiseless]  # remove noiseless sessions
    dset = dset[dset.outcome != 'abort']  # remove movement aborted trials

    return dset


def split_data(dset, fractions, seed):
    """split dataset in train/val/test folds according to fractions"""

    idx_train, idx_test = train_test_split(
        dset.index, test_size=fractions[0], random_state=seed,
        stratify=dset[['sig', 'mouse', 'hazard']]
    )

    # test split and validation split are the same if no validation fraction
    if len(fractions) == 1:
        idx_val = idx_test

    else:
        val_fraction = fractions[1] / (1 - fractions[0])
        idx_train, idx_val = train_test_split(
            idx_train, test_size=val_fraction, random_state=seed,
            stratify=dset.loc[idx_train, ['sig', 'mouse', 'hazard']]
        )

    dset = dset.copy()
    dset['train'], dset['val'], dset['test'] = False, False, False
    dset.loc[idx_train, 'train'] = True
    dset.loc[idx_val, 'val'] = True
    dset.loc[idx_test, 'test'] = True

    return dset


class StopOptimization(Exception):
    pass


class Logger:

    def __init__(self, name, model, dset, batch_size, patience=np.inf):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.patience = patience

        X, y = prepare_Xy(dset, model.n_lags, model.max_nt)
        self.X, self.y = np.vstack(X), np.vstack(y)

        self.logp = []
        self.best_logp = -np.inf
        self.max_iter = patience
        self.previous_time = time.time()

    def __call__(self, cnt):
        logp = 0
        for i in range(0, len(self.X), self.batch_size):
            j = min(i + self.batch_size, len(self.X))
            logp += self.model.predict_density(self.X[i:j], self.y[i:j]).sum()
        self.logp.append(logp)

        current_time = time.time()
        elapsed_time = current_time - self.previous_time
        self.previous_time = current_time

        elapsed = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        current = time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(current_time)
        )
        print(
            '{} :: elapsed {} :: {} :: {} (stop {}) :: current {} (best {})'
            .format(current, elapsed, self.name, cnt, self.max_iter,
                    self.logp[-1], self.best_logp)
        )

        has_improved = False
        if self.logp[-1] > self.best_logp:
            self.best_logp = self.logp[-1]
            self.max_iter = cnt + self.patience
            has_improved = True

        if cnt > self.max_iter:
            raise StopOptimization()

        return has_improved


# enumeration types used to define GP model and kernel options
Hazard = strenum('Hazard', 'early late split nonsplit all')
MeanType = strenum('MeanType', 'zero constant linear')
Hierarchy = strenum('Hierarchy', 'mouse hzrd')
Combination = strenum('Combination', 'add mul')
KernelInput = strenum(
    'KernelInput', 'full time logtime wtime stim proj expproj hzrd'
)
KernelType = strenum(
    'KernelType', 'RBF Linear Matern12 Matern32 Matern52 White Constant'
)


def main(result_dir, *dset_filename, hazard=Hazard.nonsplit,
         mean_type=MeanType.zero, kernels_type=(KernelType.RBF,),
         kernels_input=(KernelInput.full,),
         hierarchy=(), combination=Combination.add,
         sigma=1e-1, nproj=5, ntanh=5, nz=100, batch_size=50000, nlags=50,
         learning_rate=1e-3, max_iter=1000000, patience=10000,
         max_duration=np.inf, fractions=(0.2, 0.2), threads=0,
         logger_batch_size=100000, save_train=False, save_test=False,
         load_params=None, use_ard=False):
    """Fit a Gaussian process model to reaction time data

    :param str result_dir: directory for results files
    :param str dset_filename: reaction time dataset file
    :param Hazard hazard: hazard rate block type
    :param MeanType mean_type: Gaussian process mean function
    :param list[KernelType] kernels_type: kernels type
    :param list[KernelInput] kernels_input: kernels input
    :param list[Hierarchy] hierarchy: kernel hierarchical structure, if any
    :param Combination combination: kernels combination
    :param float sigma: standard deviation of Laplacian prior for projected
                        kernels
    :param int nproj: number of projections in projected kernels
    :param int ntanh: number of tanh functions in warped kernels
    :param int nz: number of inducing points per mouse
    :param int batch_size: size of mini-batches
    :param int nlags: number of past stimulus to include for each observation
    :param float learning_rate: Adam learning rate
    :param int max_iter: maximum number of iterations for optimization
    :param int patience: patience parameter for early stopping
    :param int max_duration: maximum time allowed for model fit in minutes
    :param list[float] fractions: validation and test fold fractions (same sets
                                  if only one value provided)
    :param float threads: limit number of threads for tensorflow-cpu
                          (0: no limit)
    :param int logger_batch_size: batch size for Logger objects
    :param bool save_train: save training set score
    :param bool save_test: save test set score
    :param str load_params: file used to initialize the GP model parameters
    :param bool use_ard: use ARD prior for projected kernels

    """

    # record all inputs
    main_inputs = locals().copy()

    # fix seed for reproducibility
    seed = 12345
    np.random.seed(seed)

    # load datasets and create splits for training
    dset = pd.concat([load_data(fname) for fname in dset_filename])
    dset['mouse_code'] = dset.mouse.astype('category').cat.codes
    dset['hazard_code'] = dset.hazard.astype('category').cat.codes

    dset = dset[dset.hazard != 'experimental']
    if hazard == Hazard.split:
        dset = dset[dset.hazard != 'nonsplit']
    elif hazard != Hazard.all:
        dset = dset[dset.hazard == hazard]

    dset = split_data(dset, fractions, seed)

    # limit multithreading in tensorflow-cpu
    if threads > 0:
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=threads,
            inter_op_parallelism_threads=threads
        )
        tf.Session(config=session_conf)

    # build the model
    model_opts = {
        'kernels_type': kernels_type,
        'kernels_input': kernels_input,
        'hierarchy': hierarchy,
        'combination': combination,
        'sigma': sigma,
        'n_proj': nproj,
        'n_tanh': ntanh,
        'n_z': nz,
        'batch_size': batch_size,
        'n_lags': nlags,
        'max_nt': dset.ys.map(len).max(),
        'mean_type': mean_type,
        'hazard': hazard
    }
    if ('proj' in kernels_input) and use_ard:
        model = build_model_ard(dset[dset.train], **model_opts)
    else:
        model = build_model(dset[dset.train], **model_opts)

    if load_params:
        model_params = dict(np.load(load_params))
        model.assign(model_params)

    # save options
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    with (result_path / 'arguments.json').open('w') as fd:
        json.dump(main_inputs, fd, indent=4, sort_keys=True)
    dset.to_pickle(result_path / 'dataset.pickle')
    np.savez(result_path / 'model_options.npz', **model_opts)

    # prepare logging objects
    logger = partial(Logger, model=model, batch_size=logger_batch_size)
    logger_train = logger('train', dset=dset[dset.train])
    logger_val = logger('val', dset=dset[dset.val], patience=patience)
    logger_test = logger('test', dset=dset[dset.test])

    n_iter_per_epoch = int(np.ceil(model.Y.shape[0] / model.Y.batch_size))
    max_time = time.time() + max_duration * 60
    best_model_path = result_path / 'model_params_best.npz'

    def callback(x):
        if time.time() > max_time:
            raise StopOptimization()
        if (x % n_iter_per_epoch) != 0:
            return
        if logger_val(x):
            session = model.enquire_session()
            np.savez(best_model_path, **model.read_values(session))
        if save_train:
            logger_train(x)
        if save_test:
            logger_test(x)

    # fit the model
    optimizer = gpflow.train.AdamOptimizer(learning_rate=learning_rate)
    try:
        optimizer.minimize(model, maxiter=max_iter, step_callback=callback)
    except StopOptimization:
        model.anchor(model.enquire_session())

    # save final params and log
    np.savez(result_path / 'model_params_last.npz',
             **model.read_values())
    np.savez(result_path / 'logger.npz', logp_train=logger_train.logp,
             logp_val=logger_val.logp, logp_test=logger_test.logp)


if __name__ == "__main__":
    defopt.run(main, short={})
