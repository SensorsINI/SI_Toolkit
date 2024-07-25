import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self):
        self.fig, axs = plt.subplots(5, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})

        self.fig.subplots_adjust(hspace=0.8)
        self.fig.canvas.manager.set_window_title('Live Plot')

        if not isinstance(axs, np.ndarray):
            axs = np.atleast_1d(axs)
        self.axs = axs

    def update_plots(self, data, header, selected_features):
        # Update the plots with the latest data
        if len(data) > 0:
            df = pd.DataFrame(data, columns=header)
            if 'time' in df.columns:
                time = df['time'].to_numpy()
            else:
                time = df.index
            colors = plt.rcParams["axes.prop_cycle"]()

            subplot_idx = 0  # you should not use enumerate as there are some None values in selected_features
            for feature in selected_features:
                color = next(colors)["color"]
                if feature in header:
                    data_row = df[feature]
                    self.update_timeline(feature, subplot_idx, time, data_row, color)
                    self.update_histogram(feature, subplot_idx, data_row, color)
                    subplot_idx += 1

    def clear_subplot(self, i):
        self.axs[i, 0].clear()
        self.axs[i, 1].clear()

    def update_timeline(self, feature, subplot_idx, time, data_row, color):
        # Update timeline plot
        try:
            self.axs[subplot_idx, 0].clear()
        except IndexError:
            print('Here')
        self.axs[subplot_idx, 0].set_title(
            f"Min={data_row.min():.3f}, Max={data_row.max():.3f}, Mean={data_row.mean():.3f}, Std={data_row.std():.5f}, N={data_row.size}",
            size=8)
        self.axs[subplot_idx, 0].plot(time, data_row, label=feature, marker='.', color=color, markersize=3,
                                      linewidth=0.2)
        self.axs[subplot_idx, 0].legend(loc='upper right')
        self.axs[subplot_idx, 0].grid(True, which='both', linestyle='-.', color='grey', linewidth=0.5)

    def update_histogram(self, feature, subplot_idx, data_row, color):
        # Update histogram plot
        self.axs[subplot_idx, 1].clear()
        self.axs[subplot_idx, 1].hist(data_row, bins=50, label=feature, color=color)
        self.axs[subplot_idx, 1].set_ylabel('Occurrences')
        self.axs[subplot_idx, 1].set_title(feature)
        self.axs[subplot_idx, 1].grid(True, which='both', linestyle='-.', color='grey', linewidth=0.5)

    def update_subplot_layout(self, selected_features):
        self.fig.clf()  # Clear the current figure
        subplots_count = max(1, len([feature for feature in selected_features if (feature is not None and feature!='None')]))
        axs = self.fig.subplots(subplots_count, 2, gridspec_kw={'width_ratios': [3, 1]})  # Create new subplots
        if not isinstance(axs, np.ndarray):
            axs = np.atleast_1d(axs)
        self.axs = axs

    @staticmethod
    def savefig(filepath):
        plt.savefig(filepath)
