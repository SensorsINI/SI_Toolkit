"""
The main file containing the code for plotting the charts of live plotter (real time data visualization).
You can either run this file to start the live plotter or live_plotter_GUI.py to start the GUI version.
The GUI version embeds this live plotter in a PyQt6 window, adding additional controls for the user.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pandas as pd
import seaborn as sns
from copy import deepcopy

from SI_Toolkit.LivePlotter.live_plotter_x_connection_handler_receiver import LivePlotter_ConnectionHandlerReceiver

sns.set()

DEFAULT_FEATURES_TO_PLOT = 'default'  # None, 'default', list of features

KEEP_SAMPLES_DEFAULT = 100  # at least 10
DEFAULT_ADDRESS = ('0.0.0.0', 6000)


class LivePlotter:
    def __init__(self, address=None, keep_samples=None, header_callback=None):

        if address is None:
            address = DEFAULT_ADDRESS
        if keep_samples is None:
            keep_samples = KEEP_SAMPLES_DEFAULT

        # Set up connection handler for incoming data
        self.connection_handler = LivePlotter_ConnectionHandlerReceiver(address)
        self.data = []
        self.header = None
        self.received = 0
        self.keep_samples = keep_samples
        self.selected_features = ['None'] * 5  # Default to 'None' for all subplots
        self.header_callback = header_callback  # Callback function for headers

        self.plotter = Plotter()
        self.fig = self.plotter.fig
        self.axs = self.plotter.axs

        self.animation = None

        self.paused = False
        self.frozen_data = None

    def animate(self, i):
        for buffer in self.connection_handler.poll_connection():
            self.process_buffer(buffer)

        if self.received >= 10:
            self.received = 0
            if not self.paused:
                self.plotter.update_plots(self.data, self.header, self.selected_features)

    def process_buffer(self, buffer):
        # Process incoming buffer based on its type
        if isinstance(buffer, list) and isinstance(buffer[0], str):
            self.header = buffer
            print(f'Header received: {self.header}')
            self.reset_liveplotter()
            if DEFAULT_FEATURES_TO_PLOT == 'default':
                filtered_header = [h for h in self.header if h != "time"]
                self.selected_features = filtered_header[:5] + ['None'] * (5 - len(filtered_header))  # Default first 5 headers
            elif DEFAULT_FEATURES_TO_PLOT is not None:
                self.selected_features = [feature for feature in DEFAULT_FEATURES_TO_PLOT if feature in self.header]
                self.selected_features = self.selected_features + ['None'] * (5 - len(self.selected_features))
            else:
                self.selected_features = ['None'] * 5

            # Update the subplot layout based on selected features
            self.update_subplot_layout()

            if self.header_callback:
                self.header_callback(self.header, self.selected_features)  # Call the callback with the new headers
        elif isinstance(buffer, np.ndarray):
            self.data.append(buffer)
            self.received += 1
            self.data = self.data[-self.keep_samples:]
        elif buffer == 'reset':
            self.reset_liveplotter()
        elif buffer == 'pause/resume':
            self.pause_and_resume_liveplotter()
        elif buffer == 'save' and self.header is not None:
            self.save_data()
        elif isinstance(buffer, str) and buffer == 'complete':
            print('All data received.')

    def pause_and_resume_liveplotter(self):
        # Raise flag
        self.paused = not self.paused
        # Make copy of the data - this is the data which will be saved to CSV and PDF if needed
        if self.paused:
            self.frozen_data = deepcopy(self.data)

    def reset_liveplotter(self):
        self.data = []
        print('\nLive Plot Reset\n\n\n\n')

    def save_data(self):
        # Save the current data to CSV and PDF
        filepath = 'LivePlot' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S'))
        data = self.data if not self.paused else self.frozen_data
        df = pd.DataFrame(data, columns=self.header)
        df.to_csv(filepath + '.csv', index=False)
        self.plotter.savefig(filepath + '.pdf')
        print(f'\nLive Plot saved: {filepath}.pdf')
        print(f'Live Data saved: {filepath}.csv\n\n\n\n')

    def set_keep_samples(self, keep_samples):
        self.keep_samples = keep_samples

    def update_selected_features(self, features):
        self.selected_features = features
        self.update_subplot_layout()  # Update the subplot layout based on new selected features

    def update_subplot_layout(self):
        self.plotter.update_subplot_layout(self.selected_features)

    def run_standalone(self):
        self.animation = animation.FuncAnimation(self.fig, self.animate, interval=200)
        plt.show()
        print('Finished')


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


if __name__ == '__main__':
    liveplotter = LivePlotter()
    liveplotter.run_standalone()
