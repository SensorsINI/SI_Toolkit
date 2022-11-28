
# region Imports and setting matplotlib backend

# Import functions from PyQt6 module (creating GUI)

from PyQt6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, \
    QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox, \
    QComboBox, QSlider, QFrame, QButtonGroup, QRadioButton
from PyQt6.QtCore import Qt



# Import matplotlib
# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm
if get_backend() != 'module://backend_interagg':
    use('Agg')

# Some more functions needed for interaction of matplotlib with PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import colors

# Other imports for GUI
import sys
import numpy as np

try:
    # pass
    from SI_Toolkit_ASF.brunton_widget_extensions import get_feature_label, convert_units_inplace
except ModuleNotFoundError or ImportError:
    print('Application specific extension to Brunton widget not found.')
    # raise ImportError

# endregion

# region Set color map for the plots
cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

# endregion

def run_test_gui(titles, ground_truth, predictions_list, time_axis, shift_labels):
    # Creat an instance of PyQt6 application
    # Every PyQt6 application has to contain this line
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    # Create an instance of the GUI window.
    window = MainWindow(titles, ground_truth, predictions_list, time_axis, shift_labels)
    window.show()
    # Next line hands the control over to Python GUI
    sys.exit(app.exec())

# Class implementing the main window of CartPole GUI
class MainWindow(QMainWindow):

    def __init__(self,
                 titles,
                 ground_truth,
                 predictions_list,
                 time_axis,
                 shift_labels,
                 *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.titles = titles
        self.ground_truth = ground_truth   # First element of the list is the dataset, second the columns names
        self.predictions_list = predictions_list
        self.time_axis = time_axis

        try:
            convert_units_inplace(ground_truth, self.predictions_list)
        except NameError:
            print('Function for units conversion not available.')

        self.dataset = None
        self.features_labels_dict = {}
        self.features = None
        self.feature_to_display = None

        self.shift_labels = shift_labels

        self.max_horizon = self.predictions_list[0][0].shape[-2]-1
        self.horizon = self.max_horizon//2

        self.show_all = False
        self.downsample = False
        self.current_point_at_timeaxis = (self.time_axis.shape[0]-self.max_horizon)//2
        self.select_dataset(0)

        self.MSE_at_horizon: float = 0.0
        self.sqrt_MSE_at_horizon: float = 0.0

        # region - Create container for top level layout
        layout = QVBoxLayout()
        # endregion

        # region - Change geometry of the main window
        self.setGeometry(300, 300, 2500, 1000)
        # endregion

        # region - Matplotlib figures (CartPole drawing and Slider)
        # Draw Figure
        self.fig = Figure(figsize=(25, 10))  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas = FigureCanvas(self.fig)
        self.fig.Ax = self.canvas.figure.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas, self)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.toolbar)
        lf.addWidget(self.canvas)
        layout.addLayout(lf)

        # endregion

        l_sl = QHBoxLayout()

        # region - Slider position
        l_sl_p = QVBoxLayout()
        l_sl_p.addWidget(QLabel('"Current" point in time:'))
        self.sl_p = QSlider(Qt.Orientation.Horizontal)
        self.sl_p.setMinimum(0)
        self.sl_p.setMaximum(self.time_axis.shape[0]-self.max_horizon-1)
        self.sl_p.setValue((self.time_axis.shape[0]-self.max_horizon)//2)
        self.sl_p.setTickPosition(QSlider.TickPosition.TicksBelow)
        # self.sl_p.setTickInterval(5)

        l_sl_p.addWidget(self.sl_p)
        self.sl_p.valueChanged.connect(self.slider_position_f)
        # endregion

        # region - Slider horizon
        l_sl_h = QVBoxLayout()
        l_sl_h.addWidget(QLabel('Prediction horizon:'))
        self.sl_h = QSlider(Qt.Orientation.Horizontal)
        self.sl_h.setMinimum(0)
        self.sl_h.setMaximum(self.max_horizon)
        self.sl_h.setValue(self.max_horizon//2)
        self.sl_h.setTickPosition(QSlider.TickPosition.TicksBelow)
        # self.sl_h.setTickInterval(5)
        # endregion

        l_sl_h.addWidget(self.sl_h)
        self.sl_h.valueChanged.connect(self.slider_horizon_f)

        separatorLine = QFrame()
        separatorLine.setFrameShape( QFrame.Shape.VLine )
        separatorLine.setFrameShadow( QFrame.Shadow.Raised )

        l_sl.addLayout(l_sl_p)
        l_sl.addWidget(separatorLine)
        l_sl.addLayout(l_sl_h)
        layout.addLayout(l_sl)


        # region - Define Model
        l_model = QHBoxLayout()

        # region Radio buttons to chose the model

        self.rbs_datasets = []

        for title in self.titles:
            self.rbs_datasets.append(QRadioButton(title))

        # Ensures that radio buttons are exclusive
        self.datasets_buttons_group = QButtonGroup()
        for button in self.rbs_datasets:
            self.datasets_buttons_group.addButton(button)

        lr_d = QHBoxLayout()
        lr_d.addStretch(1)
        lr_d.addWidget(QLabel('Model:'))
        for rb in self.rbs_datasets:
            rb.clicked.connect(self.RadioButtons_detaset_selection)
            lr_d.addWidget(rb)
        lr_d.addStretch(1)

        self.rbs_datasets[0].setChecked(True)

        l_model.addLayout(lr_d)

        # Add MSE at horizon
        self.lab_MSE = QLabel('sqrt(MSE) at horizon:')
        l_model.addWidget(self.lab_MSE)

        layout.addLayout(l_model)

        # endregion

        l_cb = QHBoxLayout()

        # region -- Checkbox: Show all
        self.cb_show_all = QCheckBox('Show all', self)
        if self.show_all:
            self.cb_show_all.toggle()
        self.cb_show_all.toggled.connect(self.cb_show_all_f)
        l_cb.addWidget(self.cb_show_all)
        # endregion

        # region -- Checkbox: Downsample predictions
        self.cb_downsample = QCheckBox('Downsample predictions (X2)', self)
        if self.downsample:
            self.cb_downsample.toggle()
        self.cb_downsample.toggled.connect(self.cb_downsample_f)
        l_cb.addWidget(self.cb_downsample)
        # endregion

        l_cb.addStretch(1)

        # region -- Combobox: Select feature to plot
        l_cb.addWidget(QLabel('Feature to plot:'))
        self.cb_select_feature = QComboBox()
        self.cb_select_feature.addItems(self.features_labels_dict.values())
        self.cb_select_feature.currentIndexChanged.connect(self.cb_select_feature_f)
        self.cb_select_feature.setCurrentText(self.features[0])
        l_cb.addWidget(self.cb_select_feature)

        # region - Add checkboxes to layout
        layout.addLayout(l_cb)
        # endregion

        # endregion

        # region - QUIT button
        bq = QPushButton("QUIT")
        bq.pressed.connect(self.quit_application)
        lb = QVBoxLayout()  # Layout for buttons
        lb.addWidget(bq)
        layout.addLayout(lb)
        # endregion

        # region - Create an instance of a GUI window
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.show()
        self.setWindowTitle('Testing System Model')

        # endregion

        self.redraw_canvas()


    def slider_position_f(self, value):
        self.current_point_at_timeaxis = int(value)

        self.redraw_canvas()

    def slider_horizon_f(self, value):
        self.horizon = int(value)

        self.redraw_canvas()

    def cb_show_all_f(self, state):
        if state:
            self.show_all = True
        else:
            self.show_all = False

        self.redraw_canvas()

    def cb_downsample_f(self, state):
        if state:
            self.downsample = True
        else:
            self.downsample = False

        self.redraw_canvas()

    def select_dataset(self, i):

        self.dataset = self.predictions_list[i][0]

        self.features = self.predictions_list[i][1]

        self.features_labels_dict = {}
        try:
            for feature in self.features:
                self.features_labels_dict[feature] = get_feature_label(feature)
        except NameError:
            for feature in self.features:
                self.features_labels_dict[feature] = feature

        if self.feature_to_display not in self.features:
            self.feature_to_display = self.features[0]


    def RadioButtons_detaset_selection(self):

        for i in range(len(self.rbs_datasets)):
            if self.rbs_datasets[i].isChecked():
                self.select_dataset(i)


        self.redraw_canvas()

    def cb_select_feature_f(self):
        feature_label_to_display = self.cb_select_feature.currentText()
        self.feature_to_display = list(self.features_labels_dict.keys())[list(self.features_labels_dict.values()).index(feature_label_to_display)]
        self.redraw_canvas()

    # The actions which has to be taken to properly terminate the application
    # The method is evoked after QUIT button is pressed
    # TODO: Can we connect it somehow also the the default cross closing the application?
    #   If you find out, please correct for the same in CartPole simulator
    def quit_application(self):
        # Closes the GUI window
        self.close()
        # The standard command
        # It seems however not to be working by its own
        # I don't know how it works
        QApplication.quit()


    def redraw_canvas(self):

        self.fig.Ax.clear()

        brunton_widget(self.features, self.ground_truth, self.dataset, self.time_axis,
                       axs=self.fig.Ax,
                       current_point_at_timeaxis=self.current_point_at_timeaxis,
                       feature_to_display=self.feature_to_display,
                       max_horizon=self.max_horizon,
                       horizon=self.horizon,
                       show_all=self.show_all,
                       downsample=self.downsample,
                       shift_labels=self.shift_labels)

        # self.get_sqrt_MSE_at_horizon()
        self.lab_MSE.setText("sqrt(MSE) at horizon: {:.2f}".format(self.sqrt_MSE_at_horizon))
        self.fig.Ax.grid(color="k", linestyle="--", linewidth=0.5)
        self.canvas.draw()

    def get_sqrt_MSE_at_horizon(self):

        feature_idx, = np.where(self.features == self.feature_to_display)
        ground_truth_feature_idx, = np.where(self.ground_truth[1] == self.feature_to_display)

        if self.show_all:
            predictions_at_horizon = self.dataset[:-self.horizon, self.horizon, feature_idx]
            self.MSE_at_horizon = np.mean(
                    (self.ground_truth[0][self.horizon:, ground_truth_feature_idx] - predictions_at_horizon) ** 2)

        else:
            predictions_at_horizon = self.dataset[self.current_point_at_timeaxis, self.horizon, feature_idx]
            self.MSE_at_horizon = np.mean(
                (self.ground_truth[0][self.current_point_at_timeaxis + self.horizon, ground_truth_feature_idx] - predictions_at_horizon) ** 2)

        self.sqrt_MSE_at_horizon = np.sqrt(self.MSE_at_horizon)



def brunton_widget(features, ground_truth, predictions_array, time_axis, axs=None,
                   current_point_at_timeaxis=None,
                   feature_to_display=None,
                   max_horizon=10, horizon=None,
                   show_all=True,
                   downsample=False,
                   shift_labels=1,
                   ):

    # Start at should be done bu widget (slider)
    if current_point_at_timeaxis is None:
        current_point_at_timeaxis = ground_truth[0].shape[0]//2

    if feature_to_display is None:
        feature_to_display = features[0]

    if horizon is None:
        horizon = max_horizon

    feature_idx, = np.where(features == feature_to_display)
    ground_truth_feature_idx, = np.where(ground_truth[1] == feature_to_display)

    # Brunton Plot
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    axs.plot(time_axis, ground_truth[0][:, ground_truth_feature_idx], 'k:', label='Ground Truth', marker='.', markersize=2, linewidth=0.5)
    y_lim = axs.get_ylim()
    prediction_distance = []

    try:
        y_label = get_feature_label(feature_to_display)
    except NameError:
        y_label = feature_to_display

    axs.set_ylabel(y_label, fontsize=18)
    axs.set_xlabel('Time [s]', fontsize=18)
    for i in range(horizon):

        if not show_all:
            axs.plot(time_axis[current_point_at_timeaxis], ground_truth[0][current_point_at_timeaxis, ground_truth_feature_idx],
                     'g.', markersize=16, label='Start')
            prediction_distance.append(predictions_array[current_point_at_timeaxis, i+1, feature_idx])
            if downsample:
                if (i % 2) == 0:
                    continue

            if shift_labels == 1:
                axs.plot(time_axis[current_point_at_timeaxis+i+1], prediction_distance[i],
                            c=cmap(float(i)/max_horizon),
                            marker='.')
            elif shift_labels == 0:
                axs.plot(time_axis[current_point_at_timeaxis], prediction_distance[i],
                            c='green',
                            marker='.')

        else:
            prediction_distance.append(predictions_array[:-(i+1), i+1, feature_idx])
            if downsample:
                if (i % 2) == 0:
                    continue
            if shift_labels == 1:
                axs.plot(time_axis[i+1:], prediction_distance[i],
                            c=cmap(float(i)/max_horizon),
                            marker='.', linestyle = '')
            elif shift_labels == 0:
                axs.plot(time_axis[i:-1], prediction_distance[i],
                            c='green',
                            marker='.', linestyle = '')

    # axs.set_ylim(y_lim)
    plt.show()