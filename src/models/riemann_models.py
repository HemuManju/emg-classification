import numpy as np

import matplotlib.pyplot as plt

from pyriemann.embedding import Embedding
from pyriemann.estimation import XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import umap


def svm_tangent_space_classifier(features, labels):
    """A tangent space classifier with svm for 3 classes.

    Parameters
    ----------
    features : array
        A array of features
    labels : array
        True labels

    Returns
    -------
    sklearn classifier
        Learnt classifier.

    """
    # Construct sklearn pipeline
    clf = Pipeline([('cov_transform', Covariances('oas')),
                    ('tangent_space', TangentSpace(metric='riemann')),
                    ('svm_classify', SVC(kernel='rbf', gamma='auto'))])
    # cross validation
    clf.fit(features, labels)

    return clf


def svm_tangent_space_prediction(clf, features, true_labels):
    """Predict from learnt tangent space classifier.

    Parameters
    ----------
    clf : sklearn classifier
        Learnt sklearn classifier.
    features : array
        A array of features
    true_labels : array
        True labels

    Returns
    -------
    array
        Predicted labels from the model.

    """

    # Predictions
    predictions = clf.predict(features)
    print('Classification accuracy = ', accuracy_score(true_labels,
                                                       predictions), '\n')

    return predictions


def svm_tangent_space_cross_validate(data):
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

    # Combine the dataset
    x = np.concatenate((data['train_x'], data['test_x']), axis=0)
    y = np.concatenate((data['train_y'], data['test_y']), axis=0)

    # Construct sklearn pipeline
    clf = Pipeline([('cov_transform', Covariances(estimator='lwf')),
                    ('tangent_space', TangentSpace(metric='riemann')),
                    ('svm_classify', SVC(kernel='rbf', gamma='auto'))])
    # cross validation
    scores = cross_val_score(clf, x, y, cv=KFold(5, shuffle=True))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('\n')

    return scores


def forest_tangent_space_cross_validate(data, cv=False):
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

    # Construct sklearn pipeline
    clf = Pipeline([('cov_transform', Covariances('lwf')),
                    ('tangent_space', TangentSpace(metric='riemann')),
                    ('random_forest_classify',
                     RandomForestClassifier(n_estimators=20,
                                            max_depth=10,
                                            random_state=43))])
    if cv:
        # Combine the dataset
        x = np.concatenate((data['train_x'], data['test_x']), axis=0)
        y = np.concatenate((data['train_y'], data['test_y']), axis=0)

        # cross validation
        scores = cross_val_score(clf, x, y, cv=KFold(5, shuffle=True))
        print("Accuracy: %0.4f (+/- %0.4f)" %
              (scores.mean(), scores.std() * 2))
        print('\n')
    else:
        clf = RandomForestClassifier(n_estimators=20,
                                     max_depth=10,
                                     random_state=43)
        plt.style.use('clean')
        y_train = np.argmax(data['train_y'], axis=1) + 1
        y_test = np.argmax(data['test_y'], axis=1) + 1
        classifier = clf.fit(data['train_x'], y_train)
        plot_confusion_matrix(classifier,
                              data['test_x'],
                              y_test,
                              normalize='true',
                              cmap=plt.cm.Blues)
    return None


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
    x_level_1 = x_level_1[rus.sample_indices_, :, :]
    y_level_1 = y_level_1[rus.sample_indices_].ravel()

    # Train a classifier with only this data
    clf_level_1 = Pipeline([('cov_transform', Covariances('lwf')),
                            ('tangent_space', TangentSpace(metric='riemann')),
                            ('random_forest_classify',
                             RandomForestClassifier(n_estimators=100,
                                                    random_state=43))])

    # Second level of traning
    y_level_2 = np.argmax(data['train_y'], axis=1) + 1
    idx = y_level_2 != 3
    x_level_2 = data['train_x'][idx, :, :]
    y_level_2 = y_level_2[idx].ravel()

    # Train a classifier with only this data
    clf_level_2 = Pipeline([('cov_transform', Covariances('lwf')),
                            ('tangent_space', TangentSpace(metric='riemann')),
                            ('random_forest_classify',
                             RandomForestClassifier(n_estimators=100,
                                                    random_state=43))])

    # Fir the level 2 classifier for final testing
    clf_level_1 = clf_level_1.fit(x_level_1, y_level_1)
    clf_level_2 = clf_level_2.fit(x_level_2, y_level_2)

    # Predict using first level and use the output for second level
    y_true = np.argmax(data['test_y'], axis=1) + 1
    y_pred_1 = clf_level_1.predict(data['test_x'])
    idx = y_pred_1 == 1
    y_pred_2 = clf_level_2.predict(data['test_x'][idx, :, :])
    y_pred_1[idx] = y_pred_2

    # Concatenate both of them and compare with true labels
    y_pred = y_pred_1
    score = accuracy_score(y_true, y_pred)

    return score


def forest_tangent_space_hierarchical_force(data):
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
    x_level_1 = data['train_pb'][:, 0:2, :].mean(axis=-1)
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
    x_level_2 = data['train_x'][idx, :, :]
    y_level_2 = y_level_2[idx].ravel()

    # Train a classifier with only this data
    clf_level_2 = Pipeline([('cov_transform', Covariances('lwf')),
                            ('tangent_space', TangentSpace(metric='riemann')),
                            ('random_forest_classify',
                             RandomForestClassifier(n_estimators=100,
                                                    random_state=43))])
    # scores_2 = cross_val_score(clf_level_2,
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
    y_pred_1 = clf_level_1.predict(data['test_pb'][:, 0:2, :].mean(axis=-1))
    idx = y_pred_1 == 1
    y_pred_2 = clf_level_2.predict(data['test_x'][idx, :, :])
    y_pred_1[idx] = y_pred_2

    # Concatenate both of them and compare with true labels
    y_pred = y_pred_1
    score = accuracy_score(y_true, y_pred)

    return score


def xdawn_embedding(data, use_xdawn):
    """Perform embedding of EEG data in 2D Euclidean space
    with Laplacian Eigenmaps.

    Parameters
    ----------
    data : dict
        A dictionary containing training and testing data

    Returns
    -------
    array
        Embedded

    """

    if use_xdawn:
        nfilter = 3
        xdwn = XdawnCovariances(estimator='scm', nfilter=nfilter)
        covs = xdwn.fit(data['train_x'],
                        data['train_y']).transform(data['test_x'])

        lapl = Embedding(metric='riemann', n_components=3)
        embd = lapl.fit_transform(covs)
    else:
        tangent_space = Pipeline([
            ('cov_transform', Covariances(estimator='lwf')),
            ('tangent_space', TangentSpace(metric='riemann'))
        ])
        t_space = tangent_space.fit(data['train_x'],
                                    data['train_y']).transform(data['test_x'])
        reducer = umap.UMAP(n_neighbors=30, min_dist=1, spread=2)
        embd = reducer.fit_transform(t_space)

    return embd
