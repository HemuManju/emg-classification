from pathlib import Path

import deepdish as dd
import mne
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from pyriemann.estimation import XdawnCovariances
from sklearn.model_selection import cross_val_score, train_test_split


def train_test_data(features, labels, leave_tags, config):
    """Short summary.

    Parameters
    ----------
    features : array
        An array of features.
    labels : array
        True labels.
    leave_tags : array
        An array specifying whether a subject was left out of training.
    config : yaml
        The configuration file.

    Returns
    -------
    dict
        A dict containing the train and test data.

    """
    data = {}
    labels = np.argmax(labels, axis=1)  # Convert to class int

    # Train test split
    id = np.arange(features.shape[0])
    if (leave_tags == 0).any():
        train_id = np.nonzero(tags)[0]
        test_id = np.nonzero(1 - tags)[0]
    else:
        train_id, test_id, _, _ = train_test_split(id,
                                                   id * 0,
                                                   test_size=2 *
                                                   config['TEST_SIZE'])

    # Training
    data['train_x'] = features[train_id, :, :]
    data['train_y'] = labels[train_id]

    # Testing
    data['test_x'] = features[test_id, :, :]
    data['test_y'] = labels[test_id]

    return data


def subject_pooled_data(config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels, tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Parameters
    subjects = config['subjects']
    epoch_length = config['epoch_length']
    sfreq = config['sfreq']

    # Subject information
    subjects = config['subjects']

    # Empty array (list)
    x = []
    y = []
    tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x.append(x_temp)
        y.append(y_temp)
        tags = np.concatenate((tags, y_temp[:, 0:1] * 0 + 1), axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    id = np.expand_dims(np.arange(x.shape[0]), axis=1)
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags


def subject_dependent_data(config):
    """Get subject dependent data.

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels
        2 arrays features and labels

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Parameters
    subjects = config['subjects']
    epoch_length = config['epoch_length']
    sfreq = config['sfreq']

    # Subject information
    subjects = config['subjects']
    x = np.empty((0, config['n_electrodes'], epoch_length * sfreq))
    y = np.empty((0, config['n_class']))
    tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x = np.concatenate((x, x_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)
        if subject in config['test_subjects']:
            tags = np.concatenate((tags, y_temp[:, 0:1] * 0), axis=0)
        else:
            tags = np.concatenate((tags, y_temp[:, 0:1] * 0 + 1), axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    id = np.expand_dims(np.arange(x.shape[0]), axis=1)
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :] * 1e6
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags


def subject_specific_data(subject, config):
    """Get subject specific data.

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels
        2 arrays features and labels

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Get the data
    x = data['subject_' + subject]['features']
    y = data['subject_' + subject]['labels']
    tags = y[:, 0:1] * 0 + 1

    # Balance the dataset
    rus = RandomUnderSampler()
    id = np.expand_dims(np.arange(x.shape[0]), axis=1)
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :] * 1e6
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags
