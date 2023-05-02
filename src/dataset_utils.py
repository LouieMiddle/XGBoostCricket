import os

import pandas as pd


def load_cricket_jos_buttler():
    def filter_by_pitch_x_pitch_y(data):
        data = data[(data['pitchX'] >= -2) & (data['pitchX'] <= 2)]
        data = data[(data['pitchY'] >= 0) & (data['pitchY'] <= 14)]
        return data

    def load_csv_data_mipl():
        csv_path = os.path.join("./", "mensIPLHawkeyeStats.csv")
        df = pd.read_csv(csv_path)
        df['pitchX'] = -df['pitchX']
        return df

    def boundary_mapper(run_value):
        if run_value in [0, 1]:
            return 0
        elif run_value in [4, 6]:
            return 1
        else:
            raise ValueError("Invalid batterRuns value")

    mipl_csv = load_csv_data_mipl()
    mipl_csv = filter_by_pitch_x_pitch_y(mipl_csv)

    mipl_csv = mipl_csv[
        (mipl_csv['batterRuns'] == 0) | (mipl_csv['batterRuns'] == 1) | (mipl_csv['batterRuns'] == 4) | (
                mipl_csv['batterRuns'] == 6)]
    mipl_csv['boundary'] = mipl_csv['batterRuns'].apply(boundary_mapper)

    seam = ['FAST_SEAM', 'MEDIUM_SEAM', 'SEAM']
    mipl_csv = mipl_csv[mipl_csv['batter'] == 'Jos Buttler']
    mipl_csv = mipl_csv[mipl_csv['bowlingStyle'].isin(seam)]
    mipl_csv = mipl_csv[mipl_csv['rightArmedBowl'] == True]

    categorical_attributes = []
    numerical_attributes = ['stumpsX', 'stumpsY']
    # numerical_attributes = ['stumpsX', 'stumpsY', 'pitchX', 'pitchY']
    all_columns = numerical_attributes + ['boundary']

    mipl_csv = mipl_csv[all_columns]

    features = mipl_csv.drop(['boundary'], axis=1)
    targets = mipl_csv['boundary']

    return features, targets, categorical_attributes, numerical_attributes
