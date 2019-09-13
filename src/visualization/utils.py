from pathlib import Path

import matplotlib.pyplot as plt


def get_model_path(experiment, model_number):
    """Get all the trained model paths from experiment.

    Parameters
    ----------
    experiment : str
        Which experiment trained models to load.

    Returns
    -------
    model path and model info path

    """

    read_path = str(Path(__file__).parents[2]) + '/models/' + experiment
    with open(read_path + '/time.txt', "r+") as f:
        trained_model = f.readlines()[model_number]
    model_time = trained_model.splitlines()[0]  # remove "\n"
    model_path = str(
        Path(__file__).parents[2]
    ) + '/models/' + experiment + '/model_' + model_time + '.pth'
    model_info_path = str(
        Path(__file__).parents[2]
    ) + '/models/' + experiment + '/model_info_' + model_time + '.pth'

    return model_path, model_info_path


def figure_asthetics(ax, subplot):
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """

    ax.set_axisbelow(True)
    # Hide the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Increase the ticks width
    ax.xaxis.set_tick_params(direction='out', width=1.5)
    ax.yaxis.set_tick_params(direction='out', width=1.5)

    # Increase the left and bottom spines width to match with ticks
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Increase the x and y ticks
    if not subplot:
        xtickslocs = ax.get_xticks().tolist()
        ax.set_xticks(xtickslocs[1:])
        ax.set_xticklabels(xtickslocs[1:-1])
        ytickslocs = ax.get_yticks().tolist()
        ax.set_yticks(ytickslocs)

    # Grid
    ax.grid(True)

    return None


def draw_horizontal_line(ax, value, color, linestyle, name):
    """Draws a horizontal line
    """

    ax.axhline(y=value, color=color, linestyle=linestyle, label=name)

    return None


def plot_settings():
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """

    plt.rcParams.update({'font.family': "Arial"})
    plt.rcParams.update({'font.size': 16})

    return None


def annotate_significance(ax, x1, x2, y, p):
    """Add significance annotations over a plot.

    Parameters
    ----------
    x1 : float
        x position of factor 1.
    x2 : float
        x position of factor 2.
    y : float
        Outcome variable.

    Returns
    -------
    None

    """
    h = y * 0.025
    star = []
    if p < 0.001:
        star = "***"
    elif p < 0.01:
        star = "**"
    if star:
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
        ax.text((x1 + x2) * .5,
                y,
                star,
                ha='center',
                va='bottom',
                color='k',
                size=20)

    return None
