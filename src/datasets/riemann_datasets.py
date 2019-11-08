from pathlib import Path
import random

import deepdish as dd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def get_data_split_ids(labels, leave_tags, test_size=0.15):
    """Generators training, validation, and training
    indices to be used by Dataloader.

    Parameters
    ----------
    labels : array
        An array of labels.
    test_size : float
        Test size e.g. 0.15 is 15% of whole data.

    Returns
    -------
    dict
        A dictionary of ids corresponding to train, validate, and test.

    """

    # Create an empty dictionary
    split_ids = {}

    if (leave_tags == 0).any():
        train_id = np.nonzero(leave_tags)[0]
        test_id = np.nonzero(1 - leave_tags)[0]
    else:
        ids = np.arange(labels.shape[0])
        train_id, test_id, _, _ = train_test_split(ids,
                                                   ids * 0,
                                                   test_size=2 * test_size)
    split_ids['training'] = train_id
    split_ids['testing'] = test_id

    return split_ids


def train_test_emg_data(config, leave_out=False):
    """A function to get train and test data.

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
    # Get the features and labels
    if leave_out:
        features, labels, leave_tags = subject_dependent_emg_data(config)
    else:
        features, labels, leave_tags = subject_independent_emg_data(config)

    # Get training, validation, and testing split_ids
    split_ids = get_data_split_ids(labels,
                                   leave_tags,
                                   test_size=config['TEST_SIZE'])

    labels = np.argmax(labels, axis=1)  # Convert to class int
    # Training
    data['train_x'] = features[split_ids['training'], :, :]
    data['train_y'] = labels[split_ids['training']]

    # Testing
    data['test_x'] = features[split_ids['testing'], :, :]
    data['test_y'] = labels[split_ids['testing']]

    return data


def subject_independent_emg_data(config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels, leave_tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Subject information
    subjects = config['subjects']

    # Empty array (list)
    x = []
    y = []
    leave_tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x.append(x_temp)
        y.append(y_temp)
        leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0 + 1),
                                    axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    leave_tags = leave_tags[rus.sample_indices_, :]

    return features, labels, leave_tags


def subject_dependent_emg_data(config):
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
    test_subjects = random.sample(subjects, config['n_test_subjects'])

    # Empty array (list)
    x = []
    y = []
    leave_tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x.append(x_temp)
        y.append(y_temp)
        if subject in test_subjects:
            leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0),
                                        axis=0)
        else:
            leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0 + 1),
                                        axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    leave_tags = leave_tags[rus.sample_indices_, :]

    return features, labels, leave_tags


def subject_specific_emg_data(subject, config):
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
    leave_tags = y[:, 0:1] * 0 + 1

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    leave_tags = leave_tags[rus.sample_indices_, :]

    return features, labels, leave_tags


def train_test_eeg_data(config, leave_out=False):
    """A function to get train and test data.

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
    # Get the features and labels
    if leave_out:
        features, labels, leave_tags = subject_dependent_eeg_data(config)
    else:
        features, labels, leave_tags = subject_independent_eeg_data(config)

    # Get training, validation, and testing split_ids
    split_ids = get_data_split_ids(labels,
                                   leave_tags,
                                   test_size=config['TEST_SIZE'])

    labels = np.argmax(labels, axis=1)  # Convert to class int
    # Training
    data['train_x'] = features[split_ids['training'], :, :]
    data['train_y'] = labels[split_ids['training']]

    # Testing
    data['test_x'] = features[split_ids['testing'], :, :]
    data['test_y'] = labels[split_ids['testing']]

    return data


def subject_independent_eeg_data(config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels, leave_tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    path = str(Path(__file__).parents[2] / config['clean_eeg_data'])
    data = dd.io.load(path)

    # Subject information
    subjects = config['subjects']

    # Empty array (list)
    x = []
    y = []
    leave_tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x.append(x_temp)
        y.append(y_temp)
        leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0 + 1),
                                    axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    leave_tags = leave_tags[rus.sample_indices_, :]

    return features, labels, leave_tags


def subject_dependent_eeg_data(config):
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

    path = str(Path(__file__).parents[2] / config['clean_eeg_data'])
    data = dd.io.load(path)

    # Parameters
    subjects = config['subjects']
    test_subjects = random.sample(subjects, config['n_test_subjects'])

    # Empty array (list)
    x = []
    y = []
    leave_tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x.append(x_temp)
        y.append(y_temp)
        if subject in test_subjects:
            leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0),
                                        axis=0)
        else:
            leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0 + 1),
                                        axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    leave_tags = leave_tags[rus.sample_indices_, :]

    return features, labels, leave_tags
