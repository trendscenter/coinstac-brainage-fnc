import os

import numpy as np
import pandas as pd


def form_XYMatrices_from_csv(input_dir, input_file):
    features = pd.read_csv(os.path.join(input_dir, input_file), header=None)
    X = np.array(features[features.columns[:-1]])

    y = features[features.columns[-1:]]
    y = np.array(y[y.columns])[:, 0]
    return (X, y)


def pp_metrics(*output_dicts, out_dir, out_filename, type):
    df_lists = []
    columns = [x + "_" + type for x in
               ["n_train_samples", "n_test_samples", "rmse_train", "rmse_test", "mae_train", "mae_test"]]
    columns.insert(0, "split_type")
    for output_dict in output_dicts:
        temp = [output_dict["split_type"]]
        for key in columns[1:]:
            temp.append(output_dict[key])
        df_lists.append(temp)

    df = pd.DataFrame(df_lists, columns=columns)
    save_df_to_csv(df, out_dir, out_filename)

    return df


def save_df_to_csv(df, output_dir, output_filename):
    out_file = os.path.join(output_dir, output_filename)
    print("Saving file: ", out_file)
    df.to_csv(out_file, index=False)
