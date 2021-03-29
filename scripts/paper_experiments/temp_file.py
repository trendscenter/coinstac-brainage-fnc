import os
from random import shuffle

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from preprocessor import utils as preut

COL_NAME_SUBJECT_ID = "eid"
COL_NAME_AGE = "age_when_attended_assessment_centre_f21003_2_0"


def read_fnc_gica(file_dir, file_name):
    with h5py.File(os.path.join(file_dir, file_name), 'r') as f:
        fN_data = preut.get_numpy_array(f, 'fnc_corrs_all')

    return fN_data


def read_fnc_ukbiobank(file_dir, file_name):
    with h5py.File(os.path.join(file_dir, file_name), 'r') as f:
        fN_data = preut.get_cell_array_data(f, 'fN')[0]
        icn_ins_data = preut.get_numpy_array(f, 'icn_ins')
        fnc_data = preut.get_numpy_array(f, 'corrdata')

    return fN_data, fnc_data, icn_ins_data


def generate_XYMatrices(input_dir, input_source,  data_file, label_file ):
    if input_source=="GICA":
        X, y, subj_ref_data=generate_XYMatrices_ukbiobank(input_dir, data_file, label_file)

    elif input_source=="UKBioBank_Comp2019":
        X, y, subj_ref_data=generate_XYMatrices_ukbiobank(input_dir, data_file, label_file, extract_icn_features_only=True)

    return X, y, subj_ref_data


def generate_XYMatrices_ukbiobank(input_dir, data_file, label_file, extract_icn_features_only=True):
    subj_ref_data, fnc_orig_data, icn_ins_data = read_fnc_ukbiobank(input_dir, data_file)
    df_eid = preut.get_eid_from_fN(subj_ref_data)

    df_eid_age = preut.get_eid_age_from_file(input_dir, label_file)

    df_result_eid = pd.concat([df_eid_age, df_eid], axis=1, join="inner").reindex(df_eid.index)
    y = df_result_eid[[COL_NAME_AGE]].to_numpy()

    fnc_data = fnc_orig_data

    """
    Extract only the FNC matrix corresponding to icn_ins_indexes
    """
    if extract_icn_features_only:
        icn_ins_data = icn_ins_data.astype(int).reshape((len(icn_ins_data)))
        icn_ins_data = icn_ins_data - 1  # Note: Matlab indices starts from 1 where as python starts from 0.
        fnc_data = fnc_data[:][:, icn_ins_data][:, :, icn_ins_data]

    X = preut.get_upper_diagonal_values(fnc_data)
    return (X, y, subj_ref_data)



def generate_on_server():
    dir_name = ""
    age_dir = "/data/mialab/competition2019/UKBiobank/packNship/results/scores/new/"
    age_file_name = "ukb_unaffected_clean_select_log_with_sex.tab"

    fnc_dir = "/data/mialab/competition2019/UKBiobank/FNC3/results/non-mc/"
    fnc_file_name = "data_staticFC_11754.mat"

    # [X, y, fN_data]= generate_XYMatrices_ukbiobank(dir_name, data_file=fnc_file_name, label_file=age_file_name)
    [X, y, fN_data] = generate_XYMatrices_ukbiobank(dir_name, data_file=fnc_dir + fnc_file_name,
                                                    label_file=age_dir + age_file_name)
    # preut.generate_k_splits(X, y, fN_data, save_to_dir="../test/input", k=6, shuffle_data=True)
    # aggregated_SVR(X, y)

def generate_on_desktop():
    dir_name = "/Users/sbasodi1/workplace/brain_age_pipeline/fnc/UKBioBank_Comp2019/FNC3/results/non-mc/"

    age_file_name = "ukb_unaffected_clean_select_log_with_sex.tab"

    fnc_file_name = "data_staticFC_11754.mat"

    [X, y, fN_data] = generate_XYMatrices_ukbiobank(dir_name, data_file=fnc_file_name, label_file=age_file_name)
    preut.generate_k_splits(X, y, fN_data, save_to_dir="../../test/input", k=6, shuffle_data=True,
                      type="age_range_stratified", test_size=0.1, save_splits=False)
    # aggregated_SVR(X, y)

def test_functions():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4],
                  [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([0, 1, 2, 3, 1, 1, 2, 3, 2 ,0, 1, 3, 0,3, 2 ,0, 1, 3, 0])
    y= np.array(range(len(X)))

    #preut.get_random_splits(3, X, y, shuffle_data=True, test_size=0.1)
    #preut.generate_k_splits(X, y, y, "", k=3, shuffle_data=True, type="age_range_stratified")
    preut.get_age_range_stratified_splits(2, X, y, shuffle_data=True, test_size=0.5, num_bins=4)

if __name__ == "__main__":
    #generate_on_server()
    generate_on_desktop()
    #test_functions()
