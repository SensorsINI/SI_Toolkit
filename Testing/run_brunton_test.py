# "Command line" parameters
from SI_Toolkit.Testing.Parameters_for_testing import args
from SI_Toolkit.Testing.preprocess_for_brunton import preprocess_for_brunton
from SI_Toolkit.Testing.Brunton_GUI import run_test_gui

# Predictors fron controller
from SI_Toolkit_ApplicationSpecificFiles.get_prediction_from_controller import get_prediction_for_testing_gui_from_controller


print('')
a = args()  # 'a' like arguments
print(a.__dict__)
print('')

def run_brunton_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(a)

    predictions_list = []

    # The car_controller is able to return predictions for a given prediction method
    for test_idx in range(len(a.tests)):
        predictor = a.tests[test_idx]
        predictions = get_prediction_for_testing_gui_from_controller(a, dataset, dataset_sampling_dt, predictor = predictor, dt_sampling_by_dt_fine=10 )
        predictions_list.append(predictions)

    run_test_gui(a.features, a.titles, ground_truth, predictions_list, time_axis)


if __name__=='__main__':
    run_brunton_test()