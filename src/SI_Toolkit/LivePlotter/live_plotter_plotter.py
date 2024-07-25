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

            for txt in self.fig.texts:
                txt.remove()
            self.fig.text(0.98, 0.02, f"N={df.shape[0]} samples", ha='right', va='bottom', fontsize='small')

            self.clear_timelines()
            self.clear_histograms()

            subplot_idx = 0  # you should not use enumerate as there are some None values in selected_features
            for feature in selected_features:
                color = next(colors)["color"]
                if feature in header:
                    data_row = df[feature]
                    self.update_timeline(feature, subplot_idx, time, data_row, color)
                    self.update_histogram(feature, subplot_idx, data_row, color)
                    subplot_idx += 1

    def clear_timelines(self):
        for axis in self.axs[:, 0]:
            axis.clear()
            axis.grid(True, which='both', linestyle='-.', color='grey', linewidth=0.5)

    def clear_histograms(self):
        for axis in self.axs[:, 1]:
            axis.clear()
            axis.set_ylabel('Occurrences')
            axis.grid(True, which='both', linestyle='-.', color='grey', linewidth=0.5)

    def update_timeline(self, feature, subplot_idx, time, data_row, color):
        axis = self.axs[subplot_idx, 0]
        axis.set_title(
            f"{feature}: Min={data_row.min():.3f}, Max={data_row.max():.3f}, Mean={data_row.mean():.3f}, Std={data_row.std():.5f}",
            size=8, color=color)
        axis.plot(time, data_row, label=feature, marker='.', color=color, markersize=3,
                  linewidth=0.2)
        axis.legend(loc='upper right')

    def update_histogram(self, feature, subplot_idx, data_row, color):
        # Update histogram plot
        axis = self.axs[subplot_idx, 1]
        axis.hist(data_row, bins=50, label=feature, color=color)
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def update_subplot_layout(self, selected_features):
        self.fig.clf()  # Clear the current figure
        subplots_count = max(1, len([feature for feature in selected_features if (feature is not None and feature != 'None')]))
        axs = self.fig.subplots(subplots_count, 2, gridspec_kw={'width_ratios': [3, 1]})  # Create new subplots
        if not isinstance(axs, np.ndarray):
            axs = np.atleast_1d(axs)
        self.axs = axs

    @staticmethod
    def savefig(filepath):
        plt.savefig(filepath)
