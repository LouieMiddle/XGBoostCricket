import os

import pandas as pd


def load_john_doe():
    def load_john_doe_data():
        csv_path = os.path.join("../data", "john_doe_dataset.csv")
        return pd.read_csv(csv_path)

    def boundary_mapper(run_value):
        if run_value in [0, 1]:
            return 0
        elif run_value in [4, 6]:
            return 1
        else:
            raise ValueError("Invalid batterRuns value")

    john_doe = load_john_doe_data()

    john_doe = john_doe[
        (john_doe['batterRuns'] == 0) | (john_doe['batterRuns'] == 1) | (john_doe['batterRuns'] == 4) | (
                john_doe['batterRuns'] == 6)]
    john_doe['boundary'] = john_doe['batterRuns'].apply(boundary_mapper)

    seam = ['FAST_SEAM', 'MEDIUM_SEAM', 'SEAM']
    john_doe = john_doe[john_doe['bowlingStyle'].isin(seam)]
    john_doe = john_doe[john_doe['rightArmedBowl'] == True]

    categorical_attributes = []
    # numerical_attributes = ['stumpsX', 'stumpsY']
    numerical_attributes = ['stumpsX', 'stumpsY', 'pitchX', 'pitchY']
    all_columns = numerical_attributes + ['boundary']

    john_doe = john_doe[all_columns]

    features = john_doe.drop(['boundary'], axis=1)
    targets = john_doe['boundary']

    return features, targets, categorical_attributes, numerical_attributes
