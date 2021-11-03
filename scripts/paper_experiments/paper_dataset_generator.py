import json

import numpy as np
import pandas as pd
import paper_pooled as pap_pooled

from scripts.core import preprocessor_utils as preut
from scripts.core import svr_utils as svrut


def split_local0_data():
    local_site_num = 0
    conf = json.loads(open('../../test/inputspec.json').read())[local_site_num]
    [X, y, subj_ref_data] = svrut.form_XYMatrices_with_subjects(
        input_dir=f"../../test/input/local{local_site_num}/simulatorRun/",
        input_file=conf['input_csv_path']['value'])
    y = y.reshape((y.shape[0], 1))
    preut.generate_k_partitions(X, y, subj_ref_data, save_to_dir="../../test/input/", k=1, shuffle_data=True,
                                type="age_range_stratified",
                                test_size=0.1,
                                save_partitions=True)


def merge_split_all_data_old(split_type):
    X_train, X_test, y_train, y_test, subj_train, subj_test= pap_pooled.combine_all_local_data_sep_test(split_type,
                                                                                            include_subjects=False)

    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    subj_ref_data = subj_train.extend(subj_test)

    y = y.reshape((y.shape[0], 1))
    preut.generate_k_partitions(X, y, subj_ref_data, save_to_dir="../../test/input/", k=1, shuffle_data=True,
                                type=split_type,
                                test_size=0.1,
                                save_partitions=True)


def plot_subject(fnc_upper_dia, y, subj_data):
    import matplotlib.pyplot as plt
    output_path = "/Users/sbasodi1/MEGA/work/trendz_2020/projects/brainage/brainage_fnc_rslt/paper/trends_decen_brainage/results/"

    a=fnc_upper_dia[0]
    size_X = 53
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=1)] = a
    X = X + X.T
    np.fill_diagonal(X, 1)

    plt.imshow(X, cmap='Spectral', interpolation='nearest')
    #plt.show()
    plt.savefig(output_path+"fnc_comp_map"+".png", bbox_inches = 'tight')
    plt.savefig(output_path+"fnc_comp_map"+".pdf", bbox_inches = 'tight')#, pad_inches = 0.0)



def generate_on_desktop(save_splits, plot_first_subject_comp_map=False):
    dir_name = "/Users/sbasodi1/workplace/brain_age_pipeline/fnc/UKBioBank_Comp2019/FNC3/results/non-mc/"
    dir_name = "/Users/sunitha/workplace/brainage/fnc/data/"

    age_file_name = "ukb_unaffected_clean_select_log_with_sex.tab"

    fnc_file_name = "data_staticFC_11754.mat"

    [X, y, subj_data] = svrut.form_XYMatrices_ukbiobank(dir_name, data_file=fnc_file_name, label_file=age_file_name)
    if plot_first_subject_comp_map:
        plot_subject(X, y, subj_data)
    if save_splits:
        preut.generate_k_partitions(X, y, subj_data, save_to_dir="../../test/input", k=6, shuffle_data=True,
                      split_num=0, type="random", test_size=0.1, save_partitions=save_splits)

        # aggregated_SVR(X, y)

    return X, y, subj_data


def merge_split_all_data(split_type, split_num, random_state, save_partitions=False):
    #X, y, subj_ref_data = generate_on_desktop(save_splits=False)
    X, y, subj_ref_data = pap_pooled.combine_all_local_data_with_subj()

    y = y.reshape((y.shape[0], 1))
    preut.generate_k_partitions(X, y, subj_ref_data, save_to_dir="../../test/input/", k=6, split_num=split_num,
                                shuffle_data=True,
                                type=split_type,
                                test_size=0.1,
                                random_state=random_state,
                                save_partitions=save_partitions)


def merge_split_all_data_for_sites(split_type, split_num, random_state, save_partitions=False, num_sites=6,
                                   remove_rejected_subj=False):
    X_all, y_all, subj_ref_data_all = pap_pooled.combine_all_local_data_with_subj()
    if remove_rejected_subj:
        subj_ref_data_list = subj_ref_data_all.tolist()
        rej_csv_df = pd.read_csv("./data/rejected.csv")
        rej_indx = [subj_ref_data_list.index('sub-' + str(val)) for val in rej_csv_df["SUBJID"].values]
        X = np.delete(X_all, rej_indx, axis=0)
        y = np.delete(y_all, rej_indx, axis=0)
        subj_ref_data = np.delete(subj_ref_data_all, rej_indx, axis=0)
    else:
        X = X_all
        y = y_all
        subj_ref_data = subj_ref_data_all

    output_root_dir = f"../../test/input-sitetest-{num_sites}/"
    y = y.reshape((y.shape[0], 1))
    preut.generate_k_partitions(X, y, subj_ref_data, save_to_dir=output_root_dir, k=num_sites, split_num=split_num,
                                shuffle_data=True,
                                type=split_type,
                                test_size=0.1,
                                save_partitions=save_partitions,
                                random_state=random_state)


def sample_data_across_sites():
    for split_type in ["random", "age_stratified", "age_range_stratified"]:
        for i in range(5):
            merge_split_all_data(split_type=split_type, split_num=i,
                                save_partitions=True, random_state=None) # random, age_stratified, age_range_stratified


def sample_data_varying_num_sites():
    for split_type in  ["random", "age_range_stratified"]: #, "age_stratified", "age_range_stratified"]:
        for site_num in range(2,11):
            for split_num in range(5):
                merge_split_all_data_for_sites(split_type=split_type, split_num=split_num, random_state=None,
                                               save_partitions=True, num_sites=site_num)


if __name__ == "__main__":
    #generate_on_desktop(save_splits=False, plot_first_subject_comp_map=True)
    #generate_on_desktop(save_splits=True, plot_first_subject_comp_map=False)

    # TODO: Run the following commented code to generate the data partitions across local clients
    #  before running paper_pooled.py
    #sample_data_across_sites()

    # TODO: Run the following commented code to generate the data partitions across local clients
    #  before running paper_pooled_sites_samples.py
    #sample_data_varying_num_sites()

    print("Done!")
