import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from itertools import cycle


class Plotter:
    def __init__(self):
        self.fig, axs = plt.subplots(5, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})

        self.fig.subplots_adjust(hspace=0.8)
        self.fig.canvas.manager.set_window_title('Live Plot')

        axs = np.atleast_2d(axs)
        self.axs = axs

        self.colormap = make_colormap()

    def update_plots(self, df, selected_features):
        # Update the plots with the latest data
        if len(df) > 0:

            if 'time' not in df.columns:
                df['time'] = df.index

            colors = cycle(self.colormap)
            # colors = plt.rcParams["axes.prop_cycle"]()

            for txt in self.fig.texts:
                txt.remove()
            self.fig.text(0.98, 0.02, f"N={df.shape[0]} samples", ha='right', va='bottom', fontsize='small')

            self.clear_timelines()
            self.clear_histograms()

            subplot_idx = 0  # you should not use enumerate as there are some None values in selected_features
            for features in selected_features:
                non_empty_chart = self.update_plot(df, features, subplot_idx, colors)
                if non_empty_chart:
                    subplot_idx += 1

    def update_plot(self, df, features, subplot_idx, colors):
        time = df['time']
        header = df.columns
        non_empty_chart = False
        title = []
        for feature in features:
            color = next(colors)
            if feature in header:
                data_row = df[feature]
                self.update_timeline(feature, subplot_idx, time, data_row, color)
                self.update_histogram(feature, subplot_idx, data_row, color)
                non_empty_chart = True
                title.append(self.get_title_seqment(feature, data_row, color))
        if non_empty_chart:
            self.add_title(subplot_idx, title)
        return non_empty_chart

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
        self.axs[subplot_idx, 0].plot(time, data_row, label=feature, marker='.', color=color, markersize=3,
                  linewidth=0.2)
        self.axs[subplot_idx, 0].legend(loc='upper right')

    @staticmethod
    def get_title_seqment(feature, data_row, color):
        title_segment = (f"{feature}: Min={data_row.min():.3f}, Max={data_row.max():.3f}, "
                         f"Mean={data_row.mean():.3f}, Std={data_row.std():.5f}    ", color)
        return title_segment

    def add_title(self, subplot_idx, title):
        if len(title) == 1:
            self.axs[subplot_idx, 0].set_title(title[0][0], size=8, color='black')
        else:  # For both two and three features we print stats only for two feature due to lacking space
            ax = self.axs[subplot_idx, 0]
            ax.text(0.5, 1.15, title[0][0], color=title[0][1], ha='right', va='top', fontsize=8, transform=ax.transAxes)
            ax.text(0.5, 1.15, title[1][0], color=title[1][1], ha='left', va='top', fontsize=8, transform=ax.transAxes)

        # self.axs[subplot_idx, 0].set_title(title, size=8, color=color)

    def update_histogram(self, feature, subplot_idx, data_row, color):
        # Update histogram plot
        self.axs[subplot_idx, 1].hist(data_row, bins=50, label=feature, color=color, alpha=0.5)  # Use alpha for overlapping histograms
        self.axs[subplot_idx, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def update_subplot_layout(self, selected_features):
        self.fig.clf()  # Clear the current figure
        subplots_count = max(1, len([features for features in selected_features if any(feature != 'None' for feature in features)]))
        axs = self.fig.subplots(subplots_count, 2, gridspec_kw={'width_ratios': [3, 1]})  # Create new subplots
        axs = np.atleast_2d(axs)
        self.axs = axs

    @staticmethod
    def savefig(filepath):
        plt.savefig(filepath)


def make_colormap():
    """Building a nicer oder of colors"""
    # Get a colormap and generate colors
    cmap = get_cmap('tab20')  # You can choose any colormap, e.g., 'viridis', 'plasma', etc.

    color_map = [
        # Set 1: Red, Blue, Black
        "#FF0000", "#0000FF", "#000000",

        # Set 4: Green, Pink, Navy
        "#008000", "#000080", "#FFC0CB",

        # Set 2: Orange, Purple, Cyan
        "#FFA500", "#800080", "#00CED1",

        # Set 5: Gold, Dark Violet, Turquoise
        "#9400D3", "#FFD700", "#40E0D0",

        # Set 3: Yellow, Magenta, Teal
        "#FFD700", "#FF00FF", "#008080",

    ]

    return color_map
