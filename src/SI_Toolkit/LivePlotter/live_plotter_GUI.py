"""
This script creates a PyQt6 GUI to control the LivePlotter class.
You can also run just live_plotter.py to start the live plotter without the GUI.
You will get the plots but no control over the number of samples to keep or the features to plot.
"""

import sys

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton, QFrame

import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from SI_Toolkit.LivePlotter.live_plotter import LivePlotter


class LivePlotterGUI(QWidget):
    def __init__(self, address=None, keep_samples=None):
        super().__init__()

        self.headers = []
        self.initUI(address, keep_samples)

    def initUI(self, address=None, keep_samples=None):
        self.setWindowTitle('Live Plotter Control Panel')

        layout = QVBoxLayout()

        self.plotter = LivePlotter(address=address, keep_samples=keep_samples, header_callback=self.update_headers)
        # Embed the matplotlib figure in the PyQt6 window
        self.canvas = FigureCanvas(self.plotter.fig)
        layout.addWidget(self.canvas)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Frame to hold the controls
        self.control_frame = QFrame(self)
        self.control_layout = QVBoxLayout(self.control_frame)

        # Slider to control keep_samples
        self.label = QLabel('Number of Samples to Keep:', self.control_frame)
        self.control_layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal, self.control_frame)
        self.slider.setMinimum(10)
        self.slider.setMaximum(min(1000, self.plotter.keep_samples * 2))
        self.slider.setValue(self.plotter.keep_samples)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.update_samples)
        self.update_samples(self.slider.value())
        self.control_layout.addWidget(self.slider)

        # Dropdowns to select features
        self.feature_selectors = []
        for i in range(5):
            subplot_layout = QHBoxLayout()  # Horizontal layout for each subplot's selectors

            subplot_label = QLabel(f'Subplot {i + 1}:', self.control_frame)
            subplot_layout.addWidget(subplot_label)

            subplot_selectors = []
            for j in range(3):  # Up to 3 features per subplot
                selector = QComboBox(self.control_frame)
                selector.addItem("None")
                selector.currentIndexChanged.connect(self.update_feature_selection)
                subplot_layout.addWidget(selector)
                subplot_selectors.append(selector)
            self.control_layout.addLayout(subplot_layout)
            self.feature_selectors.append(subplot_selectors)

        layout.addWidget(self.control_frame)

        # Horizontal layout for SAVE, PAUSE/RESUME and TOGGLE buttons
        button_layout = QHBoxLayout()

        # Save button
        self.save_button = QPushButton('SAVE', self)
        self.save_button.clicked.connect(self.plotter.save_data)
        button_layout.addWidget(self.save_button)

        # Pause/Resume button
        self.pause_resume_button = QPushButton('PAUSE', self)
        self.pause_resume_button.clicked.connect(self.toggle_pause_resume)
        button_layout.addWidget(self.pause_resume_button)

        # Toggle button
        self.toggle_button = QPushButton('HIDE CONTROLS', self)
        self.toggle_button.clicked.connect(self.toggle_controls)
        button_layout.addWidget(self.toggle_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.ani = None

        self.start_animation()

        self.rescale_window(1.0)

    def update_samples(self, value):
        if self.plotter:
            self.plotter.set_keep_samples(value)

    def update_feature_selection(self):
        selected_features = [[selector.currentText() for selector in subplot_selectors]
                             for subplot_selectors in self.feature_selectors]
        self.plotter.update_selected_features(selected_features)

    def start_animation(self):
        self.ani = animation.FuncAnimation(self.plotter.fig, self.plotter.animate, interval=200)
        self.canvas.draw()

    def update_headers(self, headers, selected_features):
        self.headers = headers
        for subplot_selectors in self.feature_selectors:
            # subplot_selectors is a list of 3 features which should be plotted on one chart
            for selector in subplot_selectors:
                selector.clear()
                selector.addItem("None")
                selector.addItems(headers)
        for i, subplot_selectors in enumerate(self.feature_selectors):
            for j, selector in enumerate(subplot_selectors):
                if selected_features[i][j] in headers:
                    selector.setCurrentText(selected_features[i][j])
                else:
                    selector.setCurrentText("None")

    def resizeEvent(self, event):
        # Adjust the layout and elements on window resize
        self.canvas.draw()
        super().resizeEvent(event)

    def rescale_window(self, scale_factor):
        default_size = self.sizeHint()  # Get the recommended size for the widget
        new_size = QSize(int(default_size.width() * scale_factor), int(default_size.height() * scale_factor))
        self.resize(new_size)  # Resize the window to the new size

    def toggle_controls(self):
        # Toggle the visibility of the control frame
        if self.control_frame.isVisible():
            self.control_frame.setVisible(False)
            self.toggle_button.setText('SHOW CONTROLS')
        else:
            self.control_frame.setVisible(True)
            self.toggle_button.setText('HIDE CONTROLS')

    def toggle_pause_resume(self):
        # Call the pause/resume function
        self.plotter.pause_and_resume_liveplotter()
        # Update the button label based on the current pause state
        if self.plotter.paused:
            self.pause_resume_button.setText('RESUME')
        else:
            self.pause_resume_button.setText('PAUSE')


def run_live_plotter_gui(address=None, keep_samples=None):
    app = QApplication(sys.argv)
    gui = LivePlotterGUI(address=address, keep_samples=keep_samples)
    gui.show()  # Show the GUI in windowed mode
    sys.exit(app.exec())


if __name__ == '__main__':
    run_live_plotter_gui()


# TODO: Here removing self.setGeometry(100, 100, 800, 600) helped. Maybe in Cartpole it would help to for shifted x axis
