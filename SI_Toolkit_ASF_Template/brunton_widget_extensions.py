import numpy as np

def get_feature_label(feature):

    if feature == 'angle':
        label = "Pole's Angle [deg]"
    elif feature == 'angleD':
        label = "Pole's Angular Velocity [deg/s]"
    elif feature == 'angle_cos':
        ...
    else:
        label = feature

    return label


def convert_units_inplace(ground_truth, predictions_list):
    ground_truth_dataset, ground_truth_features = ground_truth
    # Convert ground truth
    for feature in ground_truth_features:
        feature_idx, = np.where(ground_truth_features == feature)
        if feature == 'angle':
            ground_truth_dataset[:, feature_idx] *= 180.0 / np.pi
        elif feature == 'angleD':
            ...
        else:
            pass

    # Convert predictions
    for i in range(len(predictions_list)):
        predictions_array, features, _ = predictions_list[i]
        for feature in features:
            feature_idx, = np.where(features == feature)

            if feature == 'angle':
                predictions_array[:, :, feature_idx] *= 180.0/np.pi
            elif feature == 'angleD':
                ...
            else:
                pass

            predictions_list[i][0] = predictions_array
