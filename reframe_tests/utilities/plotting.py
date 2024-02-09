import numpy as np
from matplotlib import pyplot as plt


def _plot_data(subplots, time, data, label=None, colour="r", line_style="-"):
    for i in range(data.shape[1]):
        subplots[i].plot(time, data[:, i], colour, lw=2, ls=line_style, label=label)


def plot_dataset_comparison(test_time, test_data, ref_time, ref_data, model_name):
    fig, ((ex, hx), (ey, hy), (ez, hz)) = plt.subplots(
        nrows=3,
        ncols=2,
        sharex=False,
        sharey="col",
        subplot_kw=dict(xlabel="Time [ns]"),
        figsize=(20, 10),
        facecolor="w",
        edgecolor="w",
    )

    subplots = [ex, ey, ez, hx, hy, hz]
    _plot_data(subplots, test_time, test_data, model_name)
    _plot_data(subplots, ref_time, ref_data, f"{model_name} (Ref)", "g", "--")

    ylabels = [
        "$E_x$, field strength [V/m]",
        "$H_x$, field strength [A/m]",
        "$E_y$, field strength [V/m]",
        "$H_y$, field strength [A/m]",
        "$E_z$, field strength [V/m]",
        "$H_z$, field strength [A/m]",
    ]

    x_max = max(np.max(test_time), np.max(ref_time))
    for i, ax in enumerate(fig.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, x_max)
        ax.grid()
        ax.legend()

    return fig


def plot_diffs(time, diffs, plot_min=-160):
    """Plots ...

    Args:
        time:
        diffs:
        plot_min: minimum value of difference to plot (dB). Default: -160

    Returns:
        plt: matplotlib plot object.
    """
    fig, ((ex, hx), (ey, hy), (ez, hz)) = plt.subplots(
        nrows=3,
        ncols=2,
        sharex=False,
        sharey="col",
        subplot_kw=dict(xlabel="Time [ns]"),
        figsize=(20, 10),
        facecolor="w",
        edgecolor="w",
    )
    _plot_data([ex, ey, ez, hx, hy, hz], time, diffs)

    ylabels = [
        "$E_x$, difference [dB]",
        "$H_x$, difference [dB]",
        "$E_y$, difference [dB]",
        "$H_y$, difference [dB]",
        "$E_z$, difference [dB]",
        "$H_z$, difference [dB]",
    ]

    x_max = np.max(time)
    y_max = np.max(diffs)
    for i, ax in enumerate(fig.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, x_max)
        ax.set_ylim(plot_min, y_max)
        ax.grid()

    return fig
