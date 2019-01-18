#!/usr/bin/env python3

import shutil
from pathlib import Path

import numpy as np
import defopt


def main(input_model_dir, output_model_dir):
    """Convert a FVGP model with ARD prior to PartialSVGP model

    :param str input_model_dir: input model folder
    :param str output_model_dir: ouput model folder
    """

    input_dir_path = Path(input_model_dir)
    output_dir_path = Path(output_model_dir)

    # create output folder and copy unaltered files
    output_dir_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(str(input_dir_path / 'dataset.pickle'),
                str(output_dir_path / 'dataset.pickle'))
    shutil.copy(str(input_dir_path / 'model_options.npz'),
                str(output_dir_path / 'model_options.npz'))

    # convert model parameters and save them
    model_params = np.load(input_dir_path / 'model_params_best.npz')

    def convert_param(key, value):
        new_key = key.replace('FVGP', 'PartialSVGP')
        new_value = value[0] if key.endswith('W') else value
        return new_key, new_value

    model_params = dict(
        convert_param(key, value) for key, value in model_params.items()
        if not key.startswith('PartialSVGP')
    )

    np.savez(output_dir_path / 'model_params_best.npz', **model_params)


if __name__ == "__main__":
    defopt.run(main)
