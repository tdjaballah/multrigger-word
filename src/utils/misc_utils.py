import os
import pandas as pd


def clean_data_dir(files):
    [os.remove(file) for file in files]


def transform_labels(y, map_dict):
    """
    Save figure in sample dir to visualize our generated labels
    :param y: ndarray of shape (TY, N_CLASSES)
    :return: save file png
    """
    df = pd.DataFrame(y)
    df = pd.concat([pd.DataFrame({'label': i, 'x': df.index, 'y': list(df[i])}) for i in df.columns])
    df['label'] = df['label'].map(map_dict)

    return df
