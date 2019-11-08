import collections
from pathlib import Path

import ray

import deepdish as dd
import numpy as np

from .MEMD import memd


@ray.remote
def get_epoch_emd(epochs, config):
    """Get the imf of each epochs and stack them.

    Parameters
    ----------
    epochs : mne epoch
        A mne epoch for a given trial. The trial can be HighFine, LowGross etc
    config : yaml
        The configuration file.

    Returns
    -------
    array
        An array containing the imf components of all the electrodes.
    """
    n_imf = config['n_imf']
    n_electrodes = config['n_electrodes']
    sfreq = config['sfreq']
    if config['sum_imf']:
        imf_array = np.empty([len(epochs), n_electrodes, sfreq])
    else:
        imf_array = np.empty([len(epochs), n_imf * n_electrodes, sfreq])

    n_imf = config['n_imf']
    for i, epoch in enumerate(epochs):
        imf = memd(epoch)
        if config['sum_imf']:
            imf_temp = np.sum(imf[0:n_imf, :, :], axis=0)
        else:
            temp = imf[0:n_imf, :, :]
            imf_temp = temp.reshape(temp.shape[0] * temp.shape[1],
                                    temp.shape[2],
                                    order='F')
        imf_array[i, :, :] = imf_temp
    return imf_array


def emd_features(config):
    """Get the imf components of the all the epoch in a trial

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    dict
        A dictionary containing the imf of array across subjects and trails
    """

    imf_array_data = {}
    for subject in config['subjects']:

        # Initialize ray
        if not ray.is_initialized():
            ray.init(num_cpus=12)
        path = str(Path(__file__).parents[2] / config['epoch_emg_data'])
        # Load the data
        data = dd.io.load(path, group='/' + 'subject_' + subject)
        result_ids = []
        for trial in config['trials']:
            epochs = data['emg'][trial].get_data()
            result_ids.append(get_epoch_emd.remote(epochs, config))

        # Run them in parallel
        result = ray.get(result_ids)
        imf_data = collections.defaultdict(dict)
        for i, trial in enumerate(config['trials']):
            imf_data['imf'][trial] = result[i]
        imf_array_data['subject_' + subject] = imf_data

        # Shutdown ray
        ray.shutdown()

    return imf_array_data
