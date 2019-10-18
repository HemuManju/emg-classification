import torch
import torch.nn as nn


class ShallowEMGNet(nn.Module):
    """Convolution neural network class for EMG classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT, config):
        super(ShallowEMGNet, self).__init__()
        # Configuration of EMGsignals
        self.epoch_length = config['epoch_length']
        self.s_freq = config['sfreq']
        self.n_electrodes = config['n_electrodes']

        # Network blocks
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 15), stride=1),
            nn.Conv2d(10, 10, kernel_size=(8, 8), stride=1),
            nn.BatchNorm2d(10, momentum=0.1, affine=True))
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(10, OUTPUT, kernel_size=(1, 7), stride=1),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        out = self.net_1(x)

        out = out * out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)

        out = self.net_2(out)
        out = torch.squeeze(out)

        return out


class ShiftScaleEMGNet(nn.Module):
    """Convolution neural network class for EMG classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT, config):
        super(ShiftScaleEMGNet, self).__init__()
        # Configuration of EMGsignals
        self.epoch_length = config['epoch_length']
        self.s_freq = config['sfreq']
        self.n_electrodes = config['n_electrodes']

        # Network blocks
        self.mean_net = nn.Linear(self.n_electrodes,
                                  self.n_electrodes,
                                  bias=False)
        self.std_net = nn.Linear(self.n_electrodes,
                                 self.n_electrodes,
                                 bias=False)
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 15), stride=1),
            nn.Conv2d(10, 10, kernel_size=(8, 8), stride=1))
        # nn.BatchNorm2d(10, momentum=0.1, affine=True))
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(10, OUTPUT, kernel_size=(1, 7), stride=1))

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        # The normalisation network
        shift = self.mean_net(x.mean(dim=3))
        x_shifted = x - shift[:, :, :, None]
        scale = self.std_net(x_shifted.std(dim=3))
        x_scaled = x_shifted * scale[:, :, :, None]

        # The convolution network
        out = self.net_1(x_scaled)

        out = out * out
        out = self.pool(out)

        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)

        out = self.net_2(out)
        out = torch.log_softmax(out, dim=1)
        out = torch.squeeze(out)

        return out


class ShiftScaleCovEMGNet(nn.Module):
    """Convolution neural network class for EMG classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT, config):
        super(ShiftScaleCovEMGNet, self).__init__()
        # Configuration of EMGsignals
        self.epoch_length = config['epoch_length']
        self.s_freq = config['sfreq']
        self.n_electrodes = config['n_electrodes']

        # Neural Network
        self.layer_1 = nn.Linear(36, 72, bias=False)
        self.layer_2 = nn.Linear(72, 72, bias=False)
        self.layer_3 = nn.Linear(72, 36, bias=False)
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.layer_4 = nn.Linear(36, OUTPUT, bias=False)

    def forward(self, x):
        out = self.layer_1(torch.flatten(x, start_dim=1))
        out = self.layer_2(out)
        out = self.layer_3(out)

        # out = self.dropout(out)
        out = self.layer_4(out)

        out = torch.softmax(out, dim=1)
        out = torch.squeeze(out)

        return out
