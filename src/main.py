import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from data.clean_data import clean_epoch_data
from data.create_data import (create_emg_data, create_emg_epoch,
                              create_robot_dataframe)
from data.utils import save_data

from datasets.riemann_datasets import train_test_data
from datasets.torch_datasets import (train_test_iterator,
                                     subject_specific_data)
from datasets.statistics_dataset import matlab_dataframe

from models.riemann_models import (svm_tangent_space_classifier,
                                   svm_tangent_space_cross_validate,
                                   svm_tangent_space_prediction)
from models.statistical_models import mixed_effect_model
from models.torch_networks import (ShallowERPNet, ShiftScaleERPNet)
from models.torch_models import (train_torch_model, transfer_torch_model)
from models.utils import (save_trained_pytorch_model,
                          load_trained_pytorch_model)

from visualization.visualise import (plot_average_model_accuracy, plot_bar,
                                     plot_accuracy_bar)

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_emg_data') as check, check():
    data = create_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['raw_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'create_epoch_data') as check, check():
    data = create_emg_epoch(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['epoch_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'clean_epoch_data') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['clean_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'create_statistics_dataframe') as check, check():
    data = create_robot_dataframe(config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['statistics_dataframe']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'statistical_analysis') as check, check():
    dataframe = matlab_dataframe(config)

    vars = ['task + damping', 'task * damping']

    # # Perform for total force
    # for var in vars:
    #     md_task = mixed_effect_model(dataframe,
    #                                  dependent='total_force',
    #                                  independent=var)
    # Perform for velocity
    for var in vars:
        print(var)
        md_task = mixed_effect_model(dataframe,
                                     dependent='velocity',
                                     independent=var)

with skip_run('skip', 'svm_subject_independent') as check, check():
    # Get the data
    data = train_test_data(config, leave_out=False)

    # Train the classifier and predict on test data
    clf = svm_tangent_space_classifier(data['train_x'], data['train_y'])
    svm_tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'svm_subject_dependent') as check, check():
    # Get the data
    data = train_test_data(config, leave_out=True)

    # Train the classifier and predict on test data
    clf = svm_tangent_space_classifier(data['train_x'], data['train_y'])
    svm_tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'svm_crossval_subject_independent') as check, check():
    # Get the data
    data = train_test_data(config, leave_out=False)
    svm_tangent_space_cross_validate(data)

with skip_run('skip', 'shallow_subject_independent') as check, check():

    dataset = train_test_iterator(config, leave_out=False)
    model, model_info = train_torch_model(ShallowERPNet, config, dataset)
    path = Path(__file__).parents[1] / config['trained_model_path']
    save_path = str(path)
    save_trained_pytorch_model(model, model_info, save_path, save_model=False)

with skip_run('skip', 'shallow_subject_dependent') as check, check():

    dataset = train_test_iterator(config, leave_out=True)
    model, model_info = train_torch_model(ShallowERPNet, config, dataset)
    path = Path(__file__).parents[1] / config['trained_model_path']
    save_path = str(path)
    save_trained_pytorch_model(model, model_info, save_path, save_model=True)

with skip_run('skip', 'shallow_transfer_learning') as check, check():
    trained_model = load_trained_pytorch_model('experiment_1', 0)
    data_iterator = subject_specific_data(config, ['8823', '8803', '7707'])
    transfer_torch_model(trained_model, config, data_iterator)

with skip_run('skip', 'shiftscale_subject_independent') as check, check():

    for _ in range(5):
        dataset = train_test_iterator(config, [], leave_out=False)
        model, model_info = train_torch_model(ShiftScaleERPNet, config,
                                              dataset)
        path = Path(__file__).parents[1] / config['trained_model_path']
        save_path = str(path)
        save_trained_pytorch_model(model,
                                   model_info,
                                   save_path,
                                   save_model=True)

with skip_run('skip', 'shiftscale_subject_dependent') as check, check():
    test_subjects = config['test_subjects']
    for i in range(5):
        dataset = train_test_iterator(config, test_subjects, leave_out=True)
        model, model_info = train_torch_model(ShiftScaleERPNet, config,
                                              dataset)
        path = Path(__file__).parents[1] / config['trained_model_path']
        save_path = str(path)
        save_trained_pytorch_model(model,
                                   model_info,
                                   save_path,
                                   save_model=True)

with skip_run('run', 'shiftscale_transfer_learning') as check, check():
    test_subjects = config['test_subjects']

    for i in range(5):
        trained_model = load_trained_pytorch_model('experiment_1', 1)
        # summary(trained_model, input_size=(1, 8, 200))
        data_iterator = subject_specific_data(config, test_subjects)
        trained_model, model_info = transfer_torch_model(
            trained_model, config, data_iterator)
        path = Path(__file__).parents[1] / config['transfered_model_path']
        save_path = str(path)
        save_trained_pytorch_model(trained_model,
                                   model_info,
                                   save_path,
                                   save_model=False)

with skip_run('skip', 'plot_average_accuracy') as check, check():
    plot_average_model_accuracy('experiment_2', config)
    plt.show()

with skip_run('skip', 'plot_accuracy_bar') as check, check():
    colors = ['#BC0019', '#2C69A9', '#40A43A']
    plot_config = {
        'save_plot': False,
        'file_name': 'subject-dependent-accuracy',
        'color': colors[1],
        'test_color': colors[2]
    }
    plot_accuracy_bar('experiment_1', 1, config, config['subjects'],
                      plot_config)
    plt.show()

with skip_run('skip', 'plot_bar_graph') as check, check():
    # Get the data
    dataframe = matlab_dataframe(config)

    plt.subplots(figsize=(7, 4))
    sns.set(font_scale=1.2)

    # Force
    plt.subplot(1, 2, 1)
    dependent = 'task'
    independent = 'total_force'
    plot_bar(config, dataframe, independent, dependent)

    # Velocity
    plt.subplot(1, 2, 2)
    dependent = 'task'
    independent = 'velocity'
    plot_bar(config, dataframe, independent, dependent)

    plt.tight_layout()
    plt.show()
