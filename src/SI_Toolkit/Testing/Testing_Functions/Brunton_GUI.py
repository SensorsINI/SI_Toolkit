
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
    from SI_Toolkit_ASF.ToolkitCustomization.brunton_widget_extensions import get_feature_label, convert_units_inplace, calculete_additional_metrics
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

def run_test_gui(titles, ground_truth, predictions_list, time_axis):
    # Creat an instance of PyQt6 application
    # Every PyQt6 application has to contain this line
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    # Create an instance of the GUI window.
    window = MainWindow(titles, ground_truth, predictions_list, time_axis)
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
                 *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.titles = titles
        self.ground_truth = ground_truth   # First element of the list is the dataset, second the columns names
        self.predictions_list = predictions_list
        self.time_axis = time_axis

        try:
            self.ground_truth, self.predictions_list = calculete_additional_metrics(ground_truth, self.predictions_list)
        except NameError:
            print('Function for calculating additional metrics not available.')

        try:
            convert_units_inplace(self.ground_truth, self.predictions_list)
        except NameError:
            print('Function for units conversion not available.')

        self.dataset = None
        self.features_labels_dict = {}
        self.features = None
        self.feature_to_display = None
        self.feature_to_display_2 = None
        self.forward_prediction = None

        self.dt_predictions = self.predictions_list[0][2]

        self.max_horizon = None
        self.horizon = None
        self.set_horizon()

        self.show_all = False
        self.downsample = False
        self._current_point_timeaxis_index = (self.time_axis.shape[0]-self.max_horizon)//2
        self._current_point_predictions_index = None
        self._update_predictions_index()
        self.select_dataset(0)
        self.combine_features = False

        self.MSE_along_horizon: float = 0.0
        self.sqrt_MSE_along_horizon: float = 0.0
        self.MSE_at_horizon: float = 0.0
        self.sqrt_MSE_at_horizon: float = 0.0
        self.max_error: float = 0.0

        # region - Create container for top level layout
        layout = QVBoxLayout()
        # endregion

        # region - Change geometry of the main window
        self.setGeometry(300, 300, 2500, 1000)
        # endregion

        # region - Matplotlib figures (CartPole drawing and Slider)
        # Draw Figure
        self.fig = Figure()  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas = FigureCanvas(self.fig)
        self.fig.Ax = self.canvas.figure.add_subplot(111)
        self.fig.tight_layout()
        self.fig.subplots_adjust(0.06, 0.14, 0.99, 0.9)

        self.toolbar = NavigationToolbar(self.canvas, self)

        # Attach figure to the layout
        lf = QVBoxLayout()
        lf.addWidget(self.toolbar)
        lf.addWidget(self.canvas, stretch=2)


        # Second auxiliary plot:
        self.fig2 = Figure()  # Regulates the size of Figure in inches, before scaling to window size.
        self.canvas2 = FigureCanvas(self.fig2)
        self.fig2.Ax = self.canvas2.figure.add_subplot(111)
        self.fig2.tight_layout()
        self.fig2.subplots_adjust(0.06, 0.18, 0.99, 0.9)

        # Setup synchronized zooming between the two plots
        self._syncing_axes = False  # Flag to prevent infinite recursion
        self.canvas.mpl_connect('draw_event', self._on_draw_fig1)
        self.canvas2.mpl_connect('draw_event', self._on_draw_fig2)
        self._last_xlim_fig1 = None
        self._last_xlim_fig2 = None

        self.toolbar2 = NavigationToolbar(self.canvas2, self)

        # Attach figure to the layout
        lf.addWidget(self.toolbar2)
        lf.addWidget(self.canvas2, stretch=1)
        layout.addLayout(lf)

        # endregion

        l_sl = QHBoxLayout()

        # region - Slider position
        l_sl_p = QVBoxLayout()
        l_sl_p.addWidget(QLabel('"Current" point in time:'))
        self.sl_p = QSlider(Qt.Orientation.Horizontal)

        current_moment_min = 0
        current_moment_max = self.time_axis.shape[0]-self.max_horizon-1
        current_moment_init = (self.time_axis.shape[0]-self.max_horizon)//2

        if self.dt_predictions < 0:
            current_moment_min += self.max_horizon
            current_moment_max += self.max_horizon
            current_moment_init += self.max_horizon


        self.sl_p.setMinimum(current_moment_min)
        self.sl_p.setMaximum(current_moment_max)
        self.sl_p.setValue(current_moment_init)
        self.sl_p.setTickPosition(QSlider.TickPosition.TicksBelow)
        # self.sl_p.setTickInterval(5)

        l_sl_p.addWidget(self.sl_p)
        self.sl_p.valueChanged.connect(self.slider_position_f)
        # endregion

        # region - Slider horizon
        l_sl_h = QVBoxLayout()
        self.horizon_slider_label = QLabel('Prediction horizon: {}'.format(self.horizon))
        l_sl_h.addWidget(self.horizon_slider_label)
        self.sl_h = QSlider(Qt.Orientation.Horizontal)
        self.sl_h.setMinimum(0)
        self.sl_h.setMaximum(self.max_horizon)
        self.sl_h.setValue(self.horizon)
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
        self.lab_MSE = QLabel('Error at horizon - sqrt(MSE): ,')
        l_model.addWidget(self.lab_MSE)

        # Add error at horizon
        self.lab_end = QLabel('End: ,')
        l_model.addWidget(self.lab_end)

        # Add max error along horizon
        self.lab_max = QLabel('Max: ')
        l_model.addWidget(self.lab_max)

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

        # region -- Checkbox: Combine features
        self.cb_combine_features = QCheckBox('Combine features', self)
        if self.combine_features:
            self.cb_combine_features.toggle()
        self.cb_combine_features.toggled.connect(self.cb_combine_features_f)
        l_cb.addWidget(self.cb_combine_features)
        # endregion

        l_cb.addStretch(1)

        # region -- Combobox: Select feature to plot
        l_cb.addWidget(QLabel('Feature to plot:'))
        self.cb_select_feature = QComboBox()
        self.cb_select_feature_items = self.features_labels_dict
        self.cb_select_feature.addItems(self.cb_select_feature_items.values())
        self.cb_select_feature.currentIndexChanged.connect(self.cb_select_feature_f)
        self.cb_select_feature.setCurrentText(self.features[0])
        l_cb.addWidget(self.cb_select_feature)

        # region -- Combobox: Select feature to plot -- second chart
        l_cb.addWidget(QLabel('Second feature to plot:'))
        self.cb_select_feature2 = QComboBox()
        features2 = list(self.features2_labels_dict.values())
        features2.append(None)
        self.cb_select_feature2.addItems(features2)
        self.cb_select_feature2.currentIndexChanged.connect(self.cb_select_feature2_f)
        self.cb_select_feature2.setCurrentText('')
        self.cb_select_feature2_f()
        l_cb.addWidget(self.cb_select_feature2)

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

    def _on_draw_fig1(self, event):
        """Callback when fig1 is drawn - check if xlim changed and sync to fig2 if visible."""
        if self._syncing_axes or not self.canvas2.isVisible():
            return
        
        current_xlim = self.fig.Ax.get_xlim()
        if self._last_xlim_fig1 != current_xlim:
            self._last_xlim_fig1 = current_xlim
            self._syncing_axes = True
            try:
                self.fig2.Ax.set_xlim(current_xlim)
                self.canvas2.draw_idle()
            finally:
                self._syncing_axes = False

    def _on_draw_fig2(self, event):
        """Callback when fig2 is drawn - check if xlim changed and sync to fig1 if visible."""
        if self._syncing_axes or not self.canvas2.isVisible():
            return
        
        current_xlim = self.fig2.Ax.get_xlim()
        if self._last_xlim_fig2 != current_xlim:
            self._last_xlim_fig2 = current_xlim
            self._syncing_axes = True
            try:
                self.fig.Ax.set_xlim(current_xlim)
                self.canvas.draw_idle()
            finally:
                self._syncing_axes = False

    @property
    def current_point_timeaxis_index(self):
        """Get the current point timeaxis index."""
        return self._current_point_timeaxis_index

    @current_point_timeaxis_index.setter
    def current_point_timeaxis_index(self, value):
        """Set the current point timeaxis index and automatically update predictions index."""
        self._current_point_timeaxis_index = value
        self._update_predictions_index()

    @property
    def current_point_predictions_index(self):
        """Get the current point predictions index (computed based on dt_predictions)."""
        return self._current_point_predictions_index

    def _update_predictions_index(self):
        """Update predictions index based on timeaxis index and dt_predictions."""
        if self.dt_predictions < 0:
            self._current_point_predictions_index = self._current_point_timeaxis_index - self.max_horizon - 1  # -1 because control inputs for backwards prediction needs to be shifted
        else:
            self._current_point_predictions_index = self._current_point_timeaxis_index

    def set_horizon(self):
        self.max_horizon = self.predictions_list[0][0].shape[-2] - 1 - 1  # Contains seed state hance -1; Predicts one additional step hence -1 - but the last prediction does not have control input paired with it.
        self.horizon = int(np.ceil(self.max_horizon / 2))

        if hasattr(self, 'sl_h'):
            self.sl_h.setMaximum(self.max_horizon)
            self.sl_h.setValue(self.horizon)

    def slider_position_f(self, value):
        self.current_point_timeaxis_index = int(value)

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

    def cb_combine_features_f(self, combine):
        if combine:
            self.combine_features = True
            if self.cb_select_feature2.currentText() == '':
                self.cb_select_feature2.setCurrentText('pose_y')
                self.cb_select_feature2_f()
            self.cb_select_feature.setCurrentText('pose_x')
            self.cb_select_feature_f()
            self.canvas2.hide()
            self.redraw_canvas()
        else:
            self.combine_features = False
            self.cb_select_feature2.setCurrentText('')
            self.cb_select_feature2_f()

    def cb_downsample_f(self, state):
        if state:
            self.downsample = True
        else:
            self.downsample = False

        self.redraw_canvas()

    def select_dataset(self, i):

        self.dataset = self.predictions_list[i][0]

        self.features = self.predictions_list[i][1]

        self.dt_predictions = self.predictions_list[i][2]
        self.forward_prediction = self.predictions_list[i][3] if len(self.predictions_list[i]) > 3 else None

        self.features_labels_dict = {}

        for feature in self.features:
            try:
                self.features_labels_dict[feature] = get_feature_label(feature)
            except NameError:
                self.features_labels_dict[feature] = feature

        self.features2_labels_dict = {}
        for feature in self.ground_truth[1]:
            try:
                self.features2_labels_dict[feature] = get_feature_label(feature)
            except NameError:
                self.features2_labels_dict[feature] = feature

        if self.feature_to_display not in self.features:
            self.feature_to_display = self.features[0]


    def RadioButtons_detaset_selection(self):

        for i in range(len(self.rbs_datasets)):
            if self.rbs_datasets[i].isChecked():
                self.select_dataset(i)
        self.set_horizon()

        self.redraw_canvas()

    def cb_select_feature_f(self):
        feature_label_to_display = self.cb_select_feature.currentText()
        self.feature_to_display = list(self.cb_select_feature_items.keys())[list(self.cb_select_feature_items.values()).index(feature_label_to_display)]
        self.redraw_canvas()

    def cb_select_feature2_f(self):
        if self.cb_select_feature2.currentText() != '':
            feature_label_to_display = self.cb_select_feature2.currentText()
            self.feature_to_display_2 = list(self.features2_labels_dict.keys())[
                list(self.features2_labels_dict.values()).index(feature_label_to_display)]
            if not self.combine_features:
                self.canvas2.show()
        else:
            self.feature_to_display_2 = None
            self.canvas2.hide()
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

        # Save current zoom levels (both horizontal and vertical) before clearing
        xlim_fig1 = self.fig.Ax.get_xlim()
        ylim_fig1 = self.fig.Ax.get_ylim()
        xlim_fig2 = self.fig2.Ax.get_xlim()
        ylim_fig2 = self.fig2.Ax.get_ylim()

        self.fig.Ax.clear()

        if not self.combine_features:
            brunton_widget(self.features, self.ground_truth, self.dataset, self.time_axis,
                        axs=self.fig.Ax,
                        current_point_timeaxis_index=self.current_point_timeaxis_index,
                        current_point_predictions_index=self.current_point_predictions_index,
                        feature_to_display=self.feature_to_display,
                        max_horizon=self.max_horizon,
                        horizon=self.horizon,
                        show_all=self.show_all,
                        downsample=self.downsample,
                        dt_predictions=self.dt_predictions,
                        forward_prediction=self.forward_prediction)

            self.fig2.Ax.clear()
            # brunton_widget(self.features, self.ground_truth, self.dataset, self.time_axis,
            #                axs=self.fig2.Ax,
            #                current_point_timeaxis_index=self.current_point_timeaxis_index,
            #                feature_to_display=self.feature_to_display,
            #                max_horizon=self.max_horizon,
            #                horizon=self.horizon,
            #                show_all=self.show_all,
            #                downsample=self.downsample,
            #                dt_predictions=self.dt_predictions)
            if self.feature_to_display_2 is not None:
                canvas2_plot(self.features, self.ground_truth, self.time_axis,
                            axs=self.fig2.Ax,
                            current_point_timeaxis_index=self.current_point_timeaxis_index,
                            feature_to_display=self.feature_to_display_2,
                            horizon=self.horizon,
                            show_all=self.show_all,
                            dt_predictions=self.dt_predictions,
                            )
        else:
            brunton_widget_combined(self.features, self.ground_truth, self.dataset, self.time_axis,
                        axs=self.fig.Ax,
                        current_point_timeaxis_index=self.current_point_timeaxis_index,
                        current_point_predictions_index=self.current_point_predictions_index,
                        feature_to_display_1=self.feature_to_display,
                        feature_to_display_2=self.feature_to_display_2,
                        max_horizon=self.max_horizon,
                        horizon=self.horizon,
                        show_all=self.show_all,
                        downsample=self.downsample,
                        dt_predictions=self.dt_predictions)

        self.horizon_slider_label.setText('Prediction horizon: {}'.format(self.horizon))
        self.get_sqrt_MSE_at_horizon()
        self.lab_MSE.setText("Error - Avg (sqrt(MSE) all): {:.4f},  ".format(self.sqrt_MSE_along_horizon))
        self.lab_end.setText("End (sqrt(MSE) end): {:.4f},  ".format(self.sqrt_MSE_at_horizon))
        self.lab_max.setText("Max: {:.4f}".format(self.max_error))
        
        # Restore zoom levels (both horizontal and vertical) before drawing (preserve user's zoom)
        # Only restore if the limits were previously set (not default initial values)
        if xlim_fig1 != (0.0, 1.0):  # Default matplotlib limits
            self.fig.Ax.set_xlim(xlim_fig1)
        if ylim_fig1 != (0.0, 1.0):
            self.fig.Ax.set_ylim(ylim_fig1)
        if xlim_fig2 != (0.0, 1.0) and self.canvas2.isVisible():
            self.fig2.Ax.set_xlim(xlim_fig2)
        if ylim_fig2 != (0.0, 1.0) and self.canvas2.isVisible():
            self.fig2.Ax.set_ylim(ylim_fig2)
        
        self.fig.Ax.grid(color="k", linestyle="--", linewidth=0.5)
        self.fig2.Ax.grid(color="k", linestyle="--", linewidth=0.5)
        self.canvas.draw()
        self.canvas2.draw()

    def get_sqrt_MSE_at_horizon(self):

        if self.horizon == 0:
            self.MSE_along_horizon = 0.0
            self.MSE_at_horizon = 0.0
            self.sqrt_MSE_along_horizon = 0.0
            self.sqrt_MSE_at_horizon = 0.0
            self.max_error = 0.0
            return

        labels_shift = int(np.round(self.dt_predictions / np.mean(self.time_axis[1:] - self.time_axis[:-1])))

        feature_idx, = np.where(self.features == self.feature_to_display)
        ground_truth_feature_idx, = np.where(self.ground_truth[1] == self.feature_to_display)
        feature_idx = int(feature_idx)
        ground_truth_feature_idx = int(ground_truth_feature_idx)

        ground_truth = self.ground_truth[0][:, ground_truth_feature_idx]
        predictions = self.dataset[:, :, feature_idx]

        error = self._get_error(predictions, ground_truth, labels_shift)

        if self.combine_features:
            feature_idx_2, = np.where(self.features == self.feature_to_display_2)
            ground_truth_feature_idx_2, = np.where(self.ground_truth[1] == self.feature_to_display_2)
            feature_idx_2 = int(feature_idx_2)
            ground_truth_feature_idx_2 = int(ground_truth_feature_idx_2)

            ground_truth_2 = self.ground_truth[0][:, ground_truth_feature_idx_2]
            predictions_2 = self.dataset[:, :, feature_idx_2]

            error_2 = self._get_error(predictions_2, ground_truth_2, labels_shift)

            error = np.sqrt(error ** 2 + error_2 ** 2)

        self.MSE_along_horizon = np.mean(error ** 2)
        self.MSE_at_horizon = np.mean(error[..., -1] ** 2)
        self.max_error = np.max(np.abs(error))

        # Compute the square root of the calculated MSE to get the final error metric.
        self.sqrt_MSE_at_horizon = np.sqrt(self.MSE_at_horizon)
        self.sqrt_MSE_along_horizon = np.sqrt(self.MSE_along_horizon)

    def _get_error(self, predictions, ground_truth, labels_shift):
        if self.show_all:

            predictions = predictions[:, 1:self.horizon+1]

            if labels_shift != 0:
                ground_truth = self.ground_truth_for_error_calculation_show_all(ground_truth, self.horizon, labels_shift, self.max_horizon)

        else:
            if labels_shift == 0:
                ground_truth = ground_truth[
                    self.current_point_timeaxis_index, np.newaxis]  # Just the current point but keep the dimensions
                predictions = predictions[self.current_point_timeaxis_index, 0, np.newaxis]
            else:
                if labels_shift > 0:
                    gt_min = self.current_point_timeaxis_index + labels_shift
                    gt_max = self.current_point_timeaxis_index + (self.horizon + 1) * labels_shift
                else:
                    gt_min = self.current_point_timeaxis_index + (self.horizon + 1) * labels_shift
                    gt_max = self.current_point_timeaxis_index + labels_shift

                gt_step = labels_shift

                ground_truth = ground_truth[gt_min:gt_max]
                ground_truth = ground_truth[::gt_step]
                predictions = predictions[self.current_point_predictions_index, 1:self.horizon+1]

        return predictions - ground_truth

    @staticmethod
    def ground_truth_for_error_calculation_show_all(ground_truth,
                                                    horizon,
                                                    labels_shift,
                                                    alignment_horizon):
        if labels_shift == 0:
            raise ValueError("Show-all ground truth extraction expects non-zero labels_shift.")

        if labels_shift < 0:
            ground_truth = ground_truth[1:]
            reversed_gt = MainWindow.ground_truth_for_error_calculation_show_all(
                ground_truth[::-1],
                horizon,
                -labels_shift,
                alignment_horizon,
            )
            return reversed_gt[::-1]

        gt_slices = []
        for i in range(1, horizon + 1):
            gt_slices_partial = []
            for j in range(labels_shift):
                start_idx = i * labels_shift + j
                stop_idx = ground_truth.shape[0] - (alignment_horizon - i) * labels_shift + j
                gt_slices_partial.append(ground_truth[start_idx:stop_idx:labels_shift])
            if gt_slices_partial:
                gt_slices_partial = np.dstack(gt_slices_partial).flatten()
                gt_slices.append(gt_slices_partial)

        if not gt_slices:
            return np.empty((0, horizon), dtype=ground_truth.dtype)

        return np.stack(gt_slices, axis=1)


def brunton_widget(features, ground_truth, predictions_array, time_axis, axs=None,
                   current_point_timeaxis_index=None,
                   current_point_predictions_index=None,
                   feature_to_display=None,
                   max_horizon=10, horizon=None,
                   show_all=True,
                   downsample=False,
                   dt_predictions=0.0,
                   forward_prediction=None,
                   ):

    # Start at should be done bu widget (slider)
    if current_point_timeaxis_index is None:
        current_point_timeaxis_index = ground_truth[0].shape[0]//2

    if feature_to_display is None:
        feature_to_display = features[0]

    if horizon is None:
        horizon = max_horizon

    feature_idx, = np.where(features == feature_to_display)
    ground_truth_feature_idx, = np.where(ground_truth[1] == feature_to_display)

    if ground_truth_feature_idx.size == 0:
        raise ValueError(f'Feature {feature_to_display} which is requested by predictor or network not found in testing data.')

    # Brunton Plot
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    axs.plot(time_axis, ground_truth[0][:, ground_truth_feature_idx], 'k:', label='Ground Truth', marker='.', markersize=2, linewidth=0.5)
    y_lim = axs.get_ylim()

    try:
        y_label = get_feature_label(feature_to_display)
    except NameError:
        y_label = feature_to_display

    axs.set_ylabel(y_label, fontsize=14)
    axs.set_xlabel('Time [s]', fontsize=14)
    for item in axs.get_xticklabels()+axs.get_yticklabels():
        item.set_fontsize(14)

    labels_shift = int(np.round(dt_predictions/np.mean(time_axis[1:]-time_axis[:-1])))

    prediction_to_plot = []
    if not show_all:
        axs.plot(time_axis[current_point_timeaxis_index],
                 ground_truth[0][current_point_timeaxis_index, ground_truth_feature_idx],
                 'g.', markersize=16, label='Start')
        time_axis_horizon = np.arange(horizon+1)*dt_predictions + time_axis[current_point_timeaxis_index]
        for i in range(0, horizon+1):

            prediction_to_plot.append(predictions_array[current_point_predictions_index, i, feature_idx])
            if downsample:
                if (i % 2) == 0:
                    continue

            axs.plot(time_axis[current_point_timeaxis_index + i*labels_shift],
                 ground_truth[0][current_point_timeaxis_index + i*labels_shift, ground_truth_feature_idx],
                 'b.', label='Ground truth')

            axs.plot(time_axis_horizon[i], prediction_to_plot[i],
                     c=cmap((float(i-1) / max_horizon)*0.8+0.2),
                     marker='.')

        if forward_prediction is not None and current_point_predictions_index is not None:
            forward_array, forward_features, forward_dt = forward_prediction
            if forward_array is not None and forward_features is not None and forward_dt != 0.0:
                if feature_to_display in forward_features:
                    forward_feature_idx, = np.where(forward_features == feature_to_display)
                    forward_feature_idx = int(forward_feature_idx)
                    # Always plot the full forward trajectory, not limited by current horizon
                    forward_steps = forward_array.shape[1]
                    forward_values = forward_array[current_point_predictions_index, :forward_steps, forward_feature_idx].squeeze()
                    if forward_values.ndim == 0:
                        forward_values = np.array([forward_values])
                    if labels_shift != 0:
                        # Use max_horizon instead of horizon to calculate the start position
                        # This ensures the forward trajectory always starts from the end of the backward trajectory
                        forward_start_idx = current_point_timeaxis_index + max_horizon * labels_shift
                        forward_start_idx = np.clip(forward_start_idx, 0, time_axis.shape[0]-1)
                        forward_start_time = time_axis[forward_start_idx]
                    else:
                        forward_start_time = time_axis[current_point_timeaxis_index]
                    forward_time_axis = forward_start_time + np.arange(forward_steps) * forward_dt
                    axs.plot(forward_time_axis,
                             forward_values,
                             linestyle='--',
                             marker='x',
                             color='tab:orange',
                             label='Forward recon.')

    else:

        for i in range(1, horizon+1):

            prediction_to_plot.append(predictions_array[:, i, feature_idx].squeeze())
            if dt_predictions > 0:
                time_axis_to_plot = time_axis[:-max_horizon]+i*dt_predictions
            else:
                time_axis_to_plot = time_axis[max_horizon + 1:] + i * dt_predictions
            
            if downsample:
                if (i % 2) == 0:
                    continue

            axs.plot(time_axis_to_plot, prediction_to_plot[i-1],
            c=cmap((float(i-1) / max_horizon)*0.8+0.2),
            marker='.', linestyle = '')

    axs.set_ylim(y_lim)
    plt.show()


def brunton_widget_combined(features, ground_truth, predictions_array, time_axis, axs=None,
                   current_point_timeaxis_index=None,
                   current_point_predictions_index=None,
                   feature_to_display_1=None,
                   feature_to_display_2=None,
                   max_horizon=10, horizon=None,
                   show_all=True,
                   downsample=False,
                   dt_predictions=0.0,
                   ):

    # Start at should be done by widget (slider)
    feature_idx_1, = np.where(features == feature_to_display_1)
    ground_truth_feature_idx_1, = np.where(ground_truth[1] == feature_to_display_1)

    feature_idx_2, = np.where(features == feature_to_display_2)
    ground_truth_feature_idx_2, = np.where(ground_truth[1] == feature_to_display_2)

    # Brunton Plot
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    axs.plot(ground_truth[0][:, ground_truth_feature_idx_1], ground_truth[0][:, ground_truth_feature_idx_2], 'k:', label='Ground Truth', marker='.', markersize=2, linewidth=0.5)
    y_lim = axs.get_ylim()
    x_lim = axs.get_xlim()
    prediction_to_plot_1 = []
    prediction_to_plot_2 = []

    try:
        y_label = get_feature_label(feature_to_display_2)
    except NameError:
        y_label = feature_to_display_2
    axs.set_ylabel(y_label, fontsize=14)

    try:
        x_label = get_feature_label(feature_to_display_1)
    except NameError:
        x_label = feature_to_display_1
    axs.set_xlabel(x_label, fontsize=14)
    axs.set_ylabel(y_label, fontsize=14)

    for item in axs.get_xticklabels()+axs.get_yticklabels():
        item.set_fontsize(14)

    if not show_all:
        axs.plot(ground_truth[0][current_point_timeaxis_index, ground_truth_feature_idx_1],
                 ground_truth[0][current_point_timeaxis_index, ground_truth_feature_idx_2],
                 'g.', markersize=16, label='Start')
        # time_axis_predictions = np.arange(horizon+1)*dt_predictions + time_axis[current_point_timeaxis_index]
        prediction_index = current_point_predictions_index if current_point_predictions_index is not None else current_point_timeaxis_index
        for i in range(horizon):

            prediction_to_plot_1.append(predictions_array[prediction_index, i, feature_idx_1])
            prediction_to_plot_2.append(predictions_array[prediction_index, i, feature_idx_2])
            if downsample:
                if (i % 2) == 0:
                    continue

            axs.plot(ground_truth[0][current_point_timeaxis_index + i + 1, ground_truth_feature_idx_1],
                     ground_truth[0][current_point_timeaxis_index + i + 1, ground_truth_feature_idx_2],
                     'b.', label='Ground Truth')
            axs.plot(prediction_to_plot_1[i], prediction_to_plot_2[i],
                     c=cmap((float(i) / max_horizon)*0.8+0.2),
                     marker='.')

    else:

        for i in range(horizon):
            prediction_to_plot_1.append(predictions_array[:-(i+1), i, feature_idx_1])
            prediction_to_plot_2.append(predictions_array[:-(i+1), i, feature_idx_2])
            if downsample:
                if (i % 2) == 0:
                    continue

            axs.plot(prediction_to_plot_1[i], prediction_to_plot_2[i],
                        c=cmap((float(i) / max_horizon)*0.8+0.2),
                        marker='.', linestyle = '')

    axs.set_xlim(x_lim)
    axs.set_ylim(y_lim)
    plt.show()


def canvas2_plot(features, ground_truth, time_axis, axs=None,
               current_point_timeaxis_index=None,
               feature_to_display=None,
               horizon=None,
               show_all=False,
               dt_predictions=0.0,
               ):

    # Start at should be done bu widget (slider)
    if current_point_timeaxis_index is None:
        current_point_timeaxis_index = ground_truth[0].shape[0]//2

    if feature_to_display is None:
        feature_to_display = features[0]

    ground_truth_feature_idx, = np.where(ground_truth[1] == feature_to_display)

    # Brunton Plot
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    axs.plot(time_axis, ground_truth[0][:, ground_truth_feature_idx], 'k:', label='Ground Truth', marker='.', markersize=2, linewidth=0.5)
    y_lim = axs.get_ylim()

    try:
        y_label = get_feature_label(feature_to_display)
    except NameError:
        y_label = feature_to_display

    axs.set_ylabel(y_label, fontsize=10)
    axs.set_xlabel('Time [s]', fontsize=10)

    if not show_all:

        axs.plot(time_axis[current_point_timeaxis_index],
                 ground_truth[0][current_point_timeaxis_index, ground_truth_feature_idx],
                 'g.', markersize=16, label='Start')

        if dt_predictions != 0.0:
            direction = 1 if dt_predictions > 0 else -1
            within_horizon_idx = current_point_timeaxis_index + direction * np.arange(1, horizon + 1)
            time_axis_within_horizon = time_axis[within_horizon_idx]
            feature_within_horizon = ground_truth[0][within_horizon_idx, ground_truth_feature_idx]
            axs.plot(time_axis_within_horizon, feature_within_horizon, 'r.', markersize=5, label='within horizon')
        else:
            axs.plot(time_axis[current_point_timeaxis_index],
                     ground_truth[0][current_point_timeaxis_index, ground_truth_feature_idx],
                     'r.', markersize=5, label='within horizon')

    axs.set_ylim(y_lim)
    for item in axs.get_xticklabels()+axs.get_yticklabels():
        item.set_fontsize(10)

    plt.show()