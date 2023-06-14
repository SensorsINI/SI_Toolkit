import os
import yaml
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def low_pass(data, cutoff_frequency, sampling_frequency):
    nyquist_frequency = 0.5 * sampling_frequency
    normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
    x, y = signal.butter(4, normalized_cutoff_frequency, btype='low', analog=False)
    filtered_signal = signal.lfilter(x, y, data)
    return filtered_signal


def plot_frequency_spectrum(signal, sampling_frequency):
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_frequency)
    magnitude_spectrum = np.abs(fft_values)
    plt.figure(figsize=(7, 7))
    plt.plot(frequencies, magnitude_spectrum)
    plt.ylim(0, 10)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)
    plt.savefig(str(signal) + 'plot.png')

# predictors config
config_testing = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_testing.yml'), 'r'), Loader=yaml.FullLoader)

from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

def run_brunton_test(test_hls=False):

    if not test_hls:
        try:
            test_hls = config_testing['test_hls']
        except KeyError:
            test_hls = False

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(**config_testing)

    predictions_list = []
    predictors_list = config_testing['predictors_specifications_testing']
    predictor = PredictorWrapper()
    titles = []
    for predictor_specification in predictors_list:
        if predictor_specification[:2] == 'S:':
            routine = "simple evaluation"
            predictor_specification = predictor_specification[2:]
        else:
            routine = "autoregressive"
        predictor.update_predictor_config_from_specification(predictor_specification=predictor_specification)
        predictions_list.append(get_prediction(dataset, predictor, dataset_sampling_dt, routine, **config_testing))
        #predicted_values = np.squeeze(predictions_list[0][0])
        #filtered_values = low_pass(predicted_values, 0.08, 50)
        #plot_frequency_spectrum(predicted_values, 50)
        #plot_frequency_spectrum(filtered_values, 50)
        #predictions_list[0][0] = filtered_values.reshape(len(filtered_values), 1, 1)
        titles.append(predictor_specification)
        if test_hls and predictor.predictor_type == 'neural':
            predictions_list.append(get_prediction(dataset, predictor, dataset_sampling_dt, routine, **config_testing, hls=True))
            titles.append('HLS:'+predictor_specification)

    run_test_gui(titles=titles,
                 ground_truth=ground_truth, predictions_list=predictions_list, time_axis=time_axis
                 )


if __name__ == '__main__':
    run_brunton_test()
