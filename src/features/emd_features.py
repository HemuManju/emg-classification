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


def one_hot_encode(label_length, category):
    """Generate one hot encoded value of required length and category.

    Parameters
    ----------
    label_length : int
        required lenght of the array.
    category : int
        Caterory e.g: category=2, [0, 1, 0] in 3 class system

    Returns
    -------
    array
        One hot encoded array.

    """
    y = np.zeros((label_length, len(category)))
    y[:, category.index(1)] = 1

    return y


def convert_to_array(subject, trial, config):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trial : str
        Trail e.g. HighFine, LowGross.
    config : yaml
        The configuration file.

    Returns
    -------
    array
        An array of feature (x) and lables (y)

    """

    # Read path
    path = str(Path(__file__).parents[2] / config['raw_emd_feature_data'])
    # Load the data
    data = dd.io.load(path, group='/' + 'subject_' + subject)
    x_array = data['imf'][trial]

    if trial == 'HighFine':
        category = [1, 0, 0]
    if trial == 'LowGross':
        category = [0, 1, 0]
    if (trial == 'HighGross') or (trial == 'LowFine'):
        category = [0, 0, 1]

    # In order to accomodate testing
    try:
        y_array = one_hot_encode(x_array.shape[0], category)
    except ImportError:
        y_array = np.zeros((x_array.shape[0], 3))

    return x_array, y_array


def create_imf_dataset(subjects, trials, config):
    """Create feature dataset for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialise for each subject
        x_temp = []
        y_temp = []
        for trial in trials:
            # Concatenate the data corresponding to all trials types
            x_array, y_array = convert_to_array(subject, trial, config)
            x_temp.append(x_array)
            y_temp.append(y_array)

        # Convert to array
        x_temp = np.concatenate(x_temp, axis=0)
        y_temp = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['features'] = np.float32(x_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset
