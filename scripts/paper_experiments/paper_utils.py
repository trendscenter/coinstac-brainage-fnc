import os

import numpy as np
import pandas as pd


def form_XYMatrices_from_csv(input_dir, input_file):
    features = pd.read_csv(os.path.join(input_dir, input_file), header=None)
    X = np.array(features[features.columns[:-1]])

    y = features[features.columns[-1:]]
    y = np.array(y[y.columns])[:, 0]
    return (X, y)


def pp_metrics(*output_dicts, columns, out_dir, out_filename):
    df_lists = []
    if isinstance(output_dicts[0], list):
        for output_dicts_ele in output_dicts:
            df_lists.extend([[output_dict[key] for key in columns] for output_dict in output_dicts_ele])
    elif isinstance(output_dicts[0], dict):
        df_lists.extend([[output_dict[key] for key in columns] for output_dict in output_dicts])

    df = pd.DataFrame(df_lists, columns=columns)
    save_df_to_csv(df, out_dir, out_filename)

    return df


def save_df_to_csv(df, output_dir, output_filename):
    out_file = os.path.join(output_dir, output_filename)
    print("Saving file: ", out_file)
    df.to_csv(out_file, index=False)
