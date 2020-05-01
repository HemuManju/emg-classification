import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def forest_tangent_space_hierarchical(data):
    """A cross validated tangent space classifier with svm.

    Parameters
    ----------
    data : dict
        A dictionary containing training and testing data

    Returns
    -------
    cross validated scores
        A list of cross validated scores.

    """

    # Combine two classes into one class
    x_level_1 = data['train_x']
    y_level_1 = np.argmax(data['train_y'], axis=1) + 1
    y_level_1 = np.expand_dims(y_level_1, axis=1)

    # Verify if they are balanced
    print(
        sum(y_level_1 == 1) / len(y_level_1),
        sum(y_level_1 == 2) / len(y_level_1),
        sum(y_level_1 == 3) / len(y_level_1))

    # Combine C1 and C2 classes and balance the dataset for traning
    y_level_1[y_level_1 == 2] = 1
    rus = RandomUnderSampler()
    rus.fit_resample(y_level_1, y_level_1)

    # Store them in dictionary
    x_level_1 = x_level_1[rus.sample_indices_, :]
    y_level_1 = y_level_1[rus.sample_indices_].ravel()

    # Train a classifier with only this data
    clf_level_1 = RandomForestClassifier(n_estimators=100, random_state=43)
    scores_1 = cross_val_score(clf_level_1,
                               x_level_1,
                               y_level_1,
                               cv=KFold(5, shuffle=True))
    print(scores_1)

    # Second level of traning
    y_level_2 = np.argmax(data['train_y'], axis=1) + 1
    idx = y_level_2 != 3
    x_level_2 = data['train_x'][idx, :]
    y_level_2 = y_level_2[idx].ravel()

    # Train a classifier with only this data
    clf_level_2 = RandomForestClassifier(n_estimators=100, random_state=43)
    scores_2 = cross_val_score(clf_level_2,
                               x_level_2,
                               y_level_2,
                               cv=KFold(5, shuffle=True))
    print(scores_2)

    # Fir the level 2 classifier for final testing
    clf_level_1 = clf_level_1.fit(x_level_1, y_level_1)
    clf_level_2 = clf_level_2.fit(x_level_2, y_level_2)

    # Predict using first level and use the output for second level
    y_true = np.argmax(data['test_y'], axis=1) + 1
    y_pred_1 = clf_level_1.predict(data['test_x'])
    idx = y_pred_1 == 1
    y_pred_2 = clf_level_2.predict(data['test_x'][idx, :])
    y_pred_1[idx] = y_pred_2

    # Concatenate both of them and compare with true labels
    y_pred = y_pred_1
    score = accuracy_score(y_true, y_pred)

    return score


def forest_tangent_space_hierarchical_time_emg(data_time, data_emg):
    """A cross validated tangent space classifier with svm.

    Parameters
    ----------
    data : dict
        A dictionary containing training and testing data

    Returns
    -------
    cross validated scores
        A list of cross validated scores.

    """

    # Combine two classes into one class
    x_level_1 = data_time['train_x']
    y_level_1 = np.argmax(data_time['train_y'], axis=1) + 1
    # y_level_1 = np.expand_dims(y_level_1, axis=1)

    # # Combine C1 and C2 classes and balance the dataset for traning
    # y_level_1[y_level_1 == 2] = 1
    # rus = RandomUnderSampler()
    # rus.fit_resample(y_level_1, y_level_1)

    # # Store them in dictionary
    # x_level_1 = x_level_1[rus.sample_indices_, :]
    # y_level_1 = y_level_1[rus.sample_indices_].ravel()

    # Train a classifier with only this data
    clf_level_1 = RandomForestClassifier(n_estimators=100, random_state=43)
    scores_1 = cross_val_score(clf_level_1,
                               x_level_1,
                               y_level_1,
                               cv=KFold(5, shuffle=True))
    print(scores_1)

    # Second level of traning
    y_level_2 = np.argmax(data_emg['train_y'], axis=1) + 1
    idx = y_level_2 != 3
    x_level_2 = data_emg['train_x'][idx, :, :]
    y_level_2 = y_level_2[idx].ravel()

    # Train a classifier with only this data
    clf_level_2 = Pipeline([('cov_transform', Covariances('lwf')),
                            ('tangent_space', TangentSpace(metric='riemann')),
                            ('random_forest_classify',
                             RandomForestClassifier(n_estimators=100,
                                                    random_state=43))])
    scores_2 = cross_val_score(clf_level_2,
                               x_level_2,
                               y_level_2,
                               cv=KFold(5, shuffle=True))
    print(scores_2)

    # Fir the level 2 classifier for final testing
    clf_level_1 = clf_level_1.fit(x_level_1, y_level_1)
    clf_level_2 = clf_level_2.fit(x_level_2, y_level_2)

    # Predict using first level and use the output for second level
    y_true = np.argmax(data_time['test_y'], axis=1) + 1
    y_pred_1 = clf_level_1.predict(data_time['test_x'])
    idx = y_pred_1 != 3

    y_pred_2 = clf_level_2.predict(data_emg['test_x'][idx, :, :])
    y_pred_1[idx] = y_pred_2

    # Concatenate both of them and compare with true labels
    y_pred = y_pred_1
    score = accuracy_score(y_true, y_pred)

    return score
