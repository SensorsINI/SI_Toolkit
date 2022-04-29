# "Command line" parameters
from SI_Toolkit.Testing.Parameters_for_testing import args

from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton
from SI_Toolkit.Testing.Testing_Functions.get_prediction import get_prediction
from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui

a = args()  # 'a' like arguments

def run_brunton_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(a)

    predictions_list = []
    for test_idx in range(len(a.tests)):
        predictions_list.append(
            get_prediction(a, dataset, predictor_name=a.tests[test_idx], dt=dataset_sampling_dt, intermediate_steps=10)
        )

    run_test_gui(a.features, a.titles,
                 ground_truth, predictions_list, time_axis,
                 )


if __name__ == '__main__':
    run_brunton_test()
