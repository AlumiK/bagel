import os
import utils
import logging
import numpy as np
import matplotlib.pyplot as plt

from bagel.data import KPI
from datetime import datetime
from matplotlib.figure import Figure, Axes


def _plot_abnormal(kpi: KPI, ax: Axes, series: np.ndarray, color: str):
    split_index = np.where(np.diff(series) != 0)[0] + 1
    points = np.vstack((kpi.timestamps, kpi.values)).T.reshape(-1, 2)
    segments = np.split(points, split_index)
    for i in range(len(segments) - 1):
        segments[i] = np.concatenate([segments[i], [segments[i + 1][0]]])
    if series[0] == 1:
        segments = segments[0::2]
    else:
        segments = segments[1::2]
    for line in segments:
        ax.plot([datetime.fromtimestamp(timestamp) for timestamp in line[:, 0]], line[:, 1], color)


def _plot_kpi(kpi: KPI, fig: Figure):
    ax: Axes = fig.add_subplot()
    ax.plot([datetime.fromtimestamp(timestamp) for timestamp in kpi.timestamps], kpi.values)
    _plot_abnormal(kpi=kpi, ax=ax, series=kpi.labels, color='red')
    _plot_abnormal(kpi=kpi, ax=ax, series=kpi.missing, color='orange')
    ax.set_title(kpi.name)
    ax.set_ylim(-7.5, 7.5)


def main():
    utils.mkdirs(OUTPUT)
    file_list = utils.list_file(INPUT)
    progress = utils.ProgressLogger(len(file_list))

    fig: Figure = plt.figure(figsize=(16, 4))
    for file in file_list:
        kpi = utils.load_kpi(file)
        progress.log(kpi=kpi.name)
        kpi, _, _ = kpi.standardize()

        _plot_kpi(kpi=kpi, fig=fig)
        fig.savefig(os.path.join(OUTPUT, kpi.name + '.png'))
        fig.clear()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s [%(levelname)s]] %(message)s')

    INPUT = 'data'
    OUTPUT = 'out/plot'

    main()
