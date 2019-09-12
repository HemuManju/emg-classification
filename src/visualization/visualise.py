import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.torch_datasets import subject_specific_data

from .utils import (get_model_path, figure_asthetics, annotate_significance,
                    plot_settings, draw_horizontal_line)


def plot_average_model_accuracy(experiment, config, variation=False):
    """Plots the average accuracy of the pytorch model prediction.

    Parameters
    ----------
    config: yaml file
        Configuration file with all parameters
    variation : bool
        Plot variation (std) along with mean.

    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fig, ax = plt.subplots()

    keys = ['training_accuracy', 'validation_accuracy', 'testing_accuracy']
    colors = ['#BC0019', '#2C69A9', '#40A43A']
    for i, key in enumerate(keys):
        accuracy = np.empty((0, config['NUM_EPOCHS']))
        for j in range(4):
            model_path, model_info_path = get_model_path(experiment, j)
            model_info = torch.load(model_info_path, map_location=device)
            accuracy = np.vstack((model_info[key], accuracy))
        # Calculate the average
        average = np.mean(accuracy, axis=0)
        print(average[-1])
        # Plot variation
        if variation:
            min_val = average - np.min(accuracy, axis=0)
            max_val = np.max(accuracy, axis=0) - average
            ax.fill_between(range(config['NUM_EPOCHS']),
                            average - min_val,
                            average + max_val,
                            alpha=0.25,
                            color=colors[i])
        ax.plot(range(config['NUM_EPOCHS']),
                average,
                color=colors[i],
                label='average' + ' ' + key.replace('_', ' '))

    ax.set_ylim(top=1.0)
    # Specifications

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    figure_asthetics(ax, subplot=False)  # Not needed at this point
    plt.show()

    return None


def plot_model_accuracy(experiment, config, model_number):
    """Plot training, validation, and testing acurracy.

    Parameters
    ----------
    model_path : str
        A path to saved pytorch model.

    """

    model_path, model_info_path = get_model_path(experiment, model_number)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(model_info_path, map_location=device)
    training_accuracy = model_info['training_accuracy']
    validation_accuracy = model_info['validation_accuracy']
    testing_accuracy = model_info['testing_accuracy']
    epochs = np.arange(training_accuracy.shape[0])

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(epochs, training_accuracy, color=[0.69, 0.18, 0.45, 1.00])
    ax.plot(epochs, validation_accuracy, color=[0.69, 0.69, 0.69, 1.00])
    ax.plot(epochs, testing_accuracy, color=[0.12, 0.27, 0.59, 1.00])
    ax.set_ylim(top=1.0)
    # Specifications
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    figure_asthetics(ax)

    return None


def plot_accuracy_bar(experiment, model_number, config, subjects, plot_config):

    # Convert to list object if only str is given
    if not isinstance(subjects, list):
        subjects = [subjects]

    # Make an empty dataframe and empty numpy array
    # df = pd.DataFrame(index=range(5), columns=subjects)
    data = np.empty((5, len(subjects)))

    # Load the model
    model_path, model_info_path = get_model_path(experiment, model_number)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)

    for i, subject in enumerate(subjects):
        for j in range(5):
            data_iterator = subject_specific_data(config, [subject])
            accuracy = calculate_accuracy(model, data_iterator, 'testing')
            # df.loc[j, subject] = accuracy
            data[j, i] = accuracy

    # Plot the bar graph
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plot_settings()
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = [config['subjects'].index(item) for item in subjects]
    ax.bar(positions,
           mean,
           yerr=std,
           color=plot_config['color'],
           capsize=3,
           label='Trained')

    # replace color of test subjects
    if config['test_subjects']:
        ids = [
            config['subjects'].index(item) for item in config['test_subjects']
        ]
        ax.bar(ids,
               mean[ids],
               yerr=std[ids],
               color=plot_config['test_color'],
               capsize=3,
               label='Tested')

    # Draw mean values
    print(np.mean(mean))
    # draw_horizontal_line(ax, np.mean(mean), color='k', linestyle='-')
    draw_horizontal_line(ax,
                         0.3333,
                         color='#B53941',
                         linestyle='-',
                         name='Chance')

    figure_asthetics(ax, subplot=False)
    ax.set_xticks(range(0, 12))
    ax.set_xticklabels(range(1, 13))
    ax.set_ylim((0, 1.0))
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Classification accuracy (%)')
    plt.legend(ncol=3, loc="upper right")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if plot_config['save_plot']:
        name = plot_config['file_name'].lower().replace(" ", "-")
        name = name.replace("/", " ")
        path = str(Path(__file__).parents[2] /
                   config['figure_save_path']) + '/' + name + '.pdf'
        # If the folder does not exist create it
        if not os.path.isdir(config['figure_save_path']):
            os.mkdir(config['figure_save_path'])
        plt.savefig(path, bbox_inches='tight')

    return None


def plot_bar(config, dataframe, independent, dependent):
    """Bar plot of the dataframe.

    Parameters
    ----------
    config : yaml
        The configuration file.
    dataframe : dataframe
        A pandas dataframe containing the dependent and independent data
        with group data (usually this is the subject).
    dependent : str
        A string stating which variable to use as dependent variable
    independent : str
        A string stating which variable to use as independent variable

    Returns
    -------
    None

    """

    # sns.set(font_scale=1.4)

    ax = sns.barplot(x=dependent,
                     y=independent,
                     hue='damping',
                     data=dataframe,
                     capsize=.1)

    # Add significance
    if independent == 'velocity':
        y = 0.14
        annotate_significance(0.8, 1.20, y, 0.005)
        ax.set_ylim([0, y + y * 0.1])

    else:
        # Within Fine motion
        y = 5
        annotate_significance(-0.2, 0.2, y, 0.005)
        ax.set_ylim([0, y + y * 0.1])

        # Within Gross motion
        y = 16
        annotate_significance(0.8, 1.20, y, 0.005)
        ax.set_ylim([0, y + y * 0.1])

    # Other figure information
    ax.set_ylabel(independent.replace('_', ' '))
    ax.set_xlabel('task type information')

    return None


def calculate_accuracy(model, data_iterator, key):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : pytorch object
        A pytorch dataset.
    key : str
        A key to select which dataset to evaluate

    Returns
    -------
    float
        accuracy of classification for the given key.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator[key]:
            model.eval()
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1) == y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total / length

    return accuracy.numpy()
