import numpy as np
import torch
import torch.nn as nn

from .utils import (classification_accuracy, create_model_info, visual_log,
                    weights_init)


def train_torch_model(network, config, data_iterator, new_weights=False):
    """Main function to run the optimization.

    Parameters
    ----------
    network : class
        A pytorch network class.
    config : yaml
        The configuration file.
    data_iterator : dict
        A data iterator with training, validation, and testing data
    new_weights : bool
        Whether to use new weight initialization instead of default.

    Returns
    -------
    pytorch model
        A trained pytroch model.

    """
    # Device to train the model cpu or gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device being used:', device)

    # An instance of model
    model = network(config['OUTPUT'], config).to(device)
    if new_weights:
        model.apply(weights_init)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['LEARNING_RATE'])

    # Visual logger
    visual_logger = visual_log('Task type classification')
    accuracy_log = []
    for epoch in range(config['NUM_EPOCHS']):
        for x_batch, y_batch in data_iterator['training']:
            # Send the input and labels to gpu
            x_batch = x_batch.to(device)
            y_batch = (torch.max(y_batch, dim=1)[1]).to(device)

            # Forward pass
            out_put = model(x_batch)
            loss = criterion(out_put, y_batch)

            # Backward and optimize
            optimizer.zero_grad()  # For batch gradient optimisation
            loss.backward()
            optimizer.step()

        accuracy = classification_accuracy(model, data_iterator)
        accuracy_log.append(accuracy)
        visual_logger.log(epoch, [accuracy[0], accuracy[1], accuracy[2]])

    # Add loss function info to parameter.
    model_info = create_model_info(config, str(criterion),
                                   np.array(accuracy_log))

    return model, model_info


def transfer_torch_model(trained_model, config, data_iterator):

    # Perform training after freezing the weigths
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device being used:', device)

    # Switch off the gradient by default to all parameters
    for parameter in trained_model.parameters():
        parameter.requires_grad = False

    # # Switch on the gradient update only for the last layer
    # for parameter in trained_model.net_2.parameters():
    #     parameter.requires_grad = True

    # summary(trained_model, input_size=(8, 200))

    # Train only the first layers used for noramlisation
    for parameter in trained_model.mean_net.parameters():
        parameter.requires_grad = True

    for parameter in trained_model.std_net.parameters():
        parameter.requires_grad = True

    # Replace the net with new weights
    trained_model.mean_net = nn.Linear(config['n_electrodes'],
                                       config['n_electrodes']).to(device)
    trained_model.std_net = nn.Linear(config['n_electrodes'],
                                      config['n_electrodes']).to(device)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(trained_model.parameters(),
                                 lr=config['LEARNING_RATE'])

    # Visual logger
    visual_logger = visual_log('Task type classification')
    accuracy_log = []

    for epoch in range(config['NUM_TRANSFER_EPOCHS']):
        for x_batch, y_batch in data_iterator['training']:
            # Send the input and labels to gpu
            x_batch = x_batch.to(device)
            y_batch = (torch.max(y_batch, dim=1)[1]).to(device)

            # Forward pass
            out_put = trained_model(x_batch)
            loss = criterion(out_put, y_batch)

            # Backward and optimize
            optimizer.zero_grad()  # For batch gradient optimisation
            loss.backward()
            optimizer.step()

        accuracy = classification_accuracy(trained_model, data_iterator)
        accuracy_log.append(accuracy)
        visual_logger.log(epoch, [accuracy[0], 0, accuracy[1]])

    # Add loss function info to parameter.
    model_info = create_model_info(config, str(criterion),
                                   np.array(accuracy_log))

    return trained_model, model_info
