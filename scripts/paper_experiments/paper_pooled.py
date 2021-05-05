"""
This script aggregates data from all the local sites and performs regression on the combined data.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

from scripts.core import svr_utils as svrut
from scripts.paper_experiments import paper_plots as pap_plt
from scripts.paper_experiments import paper_utils as paput

TEST_DIR = "../../test/"


def aggregated_SVR(X_train, X_test, y_train, y_test):
    """
    Performs LinearSVR on the passed data.
    """
    input_list = None
    for indx, conf in enumerate(json.loads(open(TEST_DIR + 'inputspec.json').read())):
        if input_list is None:
            conf.pop("site_data")
            conf.pop("site_label")
            input_list = conf
            break

    regr = make_pipeline(preprocessing.MinMaxScaler(),
                         LinearSVR
                         (epsilon=input_list["epsilon_local"]["value"],
                          tol=input_list["tolerance_local"]["value"],
                          C=input_list["regularization_local"]["value"],
                          loss=input_list["loss_local"]["value"],
                          fit_intercept=input_list["fit_intercept_local"]["value"],
                          intercept_scaling=input_list["intercept_scaling_local"]["value"],
                          dual=input_list["dual_local"]["value"],
                          random_state=input_list["random_state_local"]["value"],
                          max_iter=input_list["max_iterations_local"]["value"]))

    regr.fit(X_train, y_train)
    params = regr.get_params()
    svr2 = params['linearsvr']
    w = svr2.coef_
    w = np.squeeze(w)
    intercept_aggr = svr2.intercept_

    y_train_pred = regr.predict(X_train)
    train_pref = svrut.get_metrics(y_train, y_train_pred)

    y_test_pred = regr.predict(X_test)
    test_pref = svrut.get_metrics(y_test, y_test_pred)

    output_dict = {
        # "intercept_aggregated": intercept_combined.tolist(),
        # "w_aggregated": w.tolist(),
        "n_train_samples_aggregated": len(y_train),
        "n_test_samples_aggregated": len(y_test),
        "rmse_train_aggregated": float(train_pref['rmse']),
        "rmse_test_aggregated": float(test_pref['rmse']),
        "mae_train_aggregated": float(train_pref['mae']),
        "mae_test_aggregated": float(test_pref['mae']),
        "phase": "aggregated",
    }

    # print(output_dict)

    return regr, output_dict

def combine_all_local_data_with_subj():
    """
    Combines data from all the local sites.
    """
    X_all, y_all, subj_all = None, None, None
    split_type="random"
    for indx, conf in enumerate(json.loads(open(TEST_DIR + 'inputspec.json').read())):
        input_dir = TEST_DIR + f"input/local{indx}/simulatorRun/"
        [local_X_train, local_y_train] = paput.form_XYMatrices_from_csv(input_dir=os.path.join(input_dir, split_type),
                                                                        input_file=f"local{indx}_fnc_age_train.csv")
        [local_X_test, local_y_test] = paput.form_XYMatrices_from_csv(input_dir=os.path.join(input_dir, split_type),
                                                                      input_file=f"local{indx}_fnc_age_test.csv")
        local_subj_train = pd.read_csv(os.path.join(input_dir,split_type, f"local{indx}_subject_ref_filename_train.csv"),
                                          header=None).to_numpy()
        local_subj_test = pd.read_csv(os.path.join(input_dir, split_type,f"local{indx}_subject_ref_filename_test.csv"),
                                         header=None).to_numpy()

        X = np.vstack((local_X_train, local_X_test))
        y = np.hstack((local_y_train, local_y_test))
        subj=np.vstack((local_subj_train, local_subj_test))

        if X_all is None:
            X_all = X
            y_all = y
            subj_all = subj
        else:
            X_all = np.vstack((X_all, X))
            y_all = np.hstack((y_all, y))
            subj_all = np.vstack((subj_all, subj))


    return (X_all, y_all, subj_all)


def combine_all_local_data(test_site_num, split_type, split_num):
    """
    Combines data from all the local sites.
    """
    X_train, X_test, y_train, y_test = get_local_site_data(local_site_num=test_site_num, split_type=split_type,
                                                           split_num=split_num)
    inputspec = json.loads(open(TEST_DIR + 'inputspec.json').read())


    for indx, conf in enumerate(inputspec):
        if indx != test_site_num:
            input_dir = TEST_DIR + f"input/local{indx}/simulatorRun/"
            input_dir = os.path.join(TEST_DIR, f"input/local{indx}/simulatorRun/", split_type + (
                "_split_" + str(split_num) if split_num is not None else ""))

            [local_X_train, local_y_train] = paput.form_XYMatrices_from_csv(input_dir=os.path.join(input_dir, split_type),
                                                                            input_file=f"local{indx}_fnc_age_train.csv")
            [local_X_test, local_y_test] = paput.form_XYMatrices_from_csv(input_dir=os.path.join(input_dir, split_type),
                                                                          input_file=f"local{indx}_fnc_age_test.csv")


            X_train = np.vstack((X_train, local_X_train))
            X_train = np.vstack((X_train, local_X_test))
            y_train = np.hstack((y_train, local_y_train))
            y_train = np.hstack((y_train, local_y_test))

    return (X_train, X_test, y_train, y_test)


def combine_all_local_data_sep_test(split_type, split_num):
    """
    Combines data from all the local sites.
    """
    X_train, y_train, X_test, y_test = None, None, None, None
    for indx, conf in enumerate(json.loads(open(TEST_DIR + 'inputspec.json').read())):
        input_dir = TEST_DIR + f"input/local{indx}/simulatorRun/"
        input_dir = os.path.join(TEST_DIR, f"input/local{indx}/simulatorRun/", split_type + (
            "_split_" + str(split_num) if split_num is not None else ""))

        [local_X_train, local_y_train] = paput.form_XYMatrices_from_csv(input_dir=input_dir,
                                                                        input_file=f"local{indx}_fnc_age_train.csv")
        [local_X_test, local_y_test] = paput.form_XYMatrices_from_csv(input_dir=input_dir,
                                                                      input_file=f"local{indx}_fnc_age_test.csv")

        if X_train is None:
            X_train = local_X_train
            y_train = local_y_train
            X_test = local_X_test
            y_test = local_y_test
        else:
            X_train = np.vstack((X_train, local_X_train))
            y_train = np.hstack((y_train, local_y_train))
            X_test = np.vstack((X_test, local_X_test))
            y_test = np.hstack((y_test, local_y_test))

    return (X_train, X_test, y_train, y_test)


def get_local_site_data(local_site_num, split_type, split_num):
    conf = json.loads(open(TEST_DIR + 'inputspec.json').read())[local_site_num]
    input_dir = TEST_DIR + f"input/local{local_site_num}/simulatorRun/"
    input_dir = os.path.join(TEST_DIR, f"input/local{local_site_num}/simulatorRun/", split_type + (
        "_split_" + str(split_num) if split_num is not None else ""))

    [local_X_train, local_y_train] = paput.form_XYMatrices_from_csv(input_dir=input_dir,
                                                                    input_file=f"local{local_site_num}_fnc_age_train.csv")
    [local_X_test, local_y_test] = paput.form_XYMatrices_from_csv(input_dir=input_dir,
                                                                  input_file=f"local{local_site_num}_fnc_age_test.csv")


    return local_X_train, local_X_test, local_y_train, local_y_test


def perform_pca(X_train, X_test):
    """
    Performs PCA for feature reduction
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_pca = scaler.transform(X_train)
    X_test_pca = scaler.transform(X_test)

    pca = PCA(.95)
    pca.fit(X_train_pca)
    X_train_pca = pca.transform(X_train_pca)
    X_test_pca = pca.transform(X_test_pca)

    return X_train_pca, X_test_pca


def build_aggregated_model(test_site_num, split_type, split_num, pca=False, combine_all_local_tests=False):

    if combine_all_local_tests:
        X_train, X_test, y_train, y_test = combine_all_local_data_sep_test(split_type=split_type, split_num=split_num)
    else:
        X_train, X_test, y_train, y_test = combine_all_local_data(test_site_num=test_site_num, split_type=split_type,
                                                                  split_num=split_num)

    print(f"Combined train and test data from all the local clients, split_type:{split_type}, split_num: {split_num}" )
    if pca:
        print("Using PCA features..")
        X_train, X_test = perform_pca(X_train, X_test)

    print("Running SVR now.")
    model, output_dict = aggregated_SVR(X_train, X_test, y_train, y_test)
    #print(output_dict)

    # paput.pp_metrics(output_dict, split_type=split_type, out_dir=output_path, out_filename="FNC_exp_v1_aggregated.csv",
    #                 type="aggregated")
    return output_dict


def get_distributed_model(owner_node_num, split_type, split_num):
    local_output_dicts = []
    local_weights = []

    X_owner_train, X_owner_test, y_owner_train, y_owner_test = get_local_site_data(local_site_num=owner_node_num,
                                                                                   split_type=split_type,
                                                                                   split_num=split_num)
    X_all_test, y_all_test = X_owner_test, y_owner_test
    # Train local models
    for indx, conf in enumerate(json.loads(open(TEST_DIR + 'inputspec.json').read())):
        if indx != owner_node_num:
            input_dir = TEST_DIR + f"input/local{indx}/simulatorRun/"

            input_dir = os.path.join(TEST_DIR, f"input/local{indx}/simulatorRun/", split_type + (
                "_split_" + str(split_num) if split_num is not None else ""))

            [local_X_train, local_y_train] = paput.form_XYMatrices_from_csv(input_dir=input_dir,
                                                                            input_file=f"local{indx}_fnc_age_train.csv")
            [local_X_test, local_y_test] = paput.form_XYMatrices_from_csv(input_dir=input_dir,
                                                                          input_file=f"local{indx}_fnc_age_test.csv")
            # Add local test data to owner test data
            X_all_test = np.vstack((X_all_test, local_X_test))
            y_all_test = np.hstack((y_all_test, local_y_test))

            # build local model
            local_regr, local_output_dict = aggregated_SVR(local_X_train, local_X_test, local_y_train, local_y_test)

            params = local_regr.get_params()
            svr2 = params['linearsvr']
            w = svr2.coef_
            w = np.squeeze(w)
            local_output_dicts.append(local_output_dict)
            local_weights.append(w.T)

    # Train distributed model
    w_locals = np.array(local_weights)
    w_locals = w_locals.astype(np.double)

    # TODO CHECK: w_avg = np.mean(w_locals, axis=1)
    w_avg = np.mean(w_locals, axis=0)
    w_avg = w_avg.reshape(-1, 1)

    U_train = np.matmul(X_owner_train, w_avg)
    U_test = np.matmul(X_all_test, w_avg)

    owner_regr, owner_output_dict = aggregated_SVR(U_train, U_test, y_owner_train, y_all_test)

    return owner_output_dict, local_output_dicts


def build_distributed_model(owner_node_num, split_type, split_num, pca, combine_all_local_tests=False):
    print(f"Running distributed SVR now. Owner node: local{str(owner_node_num)}, split_type:{split_type}, "
          f"split_num: {split_num}" )

    owner_output_dict, local_output_dicts = get_distributed_model(owner_node_num, split_type, split_num)
    print(owner_output_dict)

    # paput.pp_metrics(output_dict, split_type=split_type, out_dir=output_path, out_filename="FNC_exp_v1_aggregated.csv",
    #                 type="aggregated")

    return owner_output_dict, local_output_dicts


def cross_validate_aggregated_model(split_types, num_splits, pca=False, combine_all_local_tests=True):
    output_filename = "FNC_aggregated_cv" + ("_sep_test.csv" if combine_all_local_tests else ".csv")
    outputs = []

    for split_type in split_types:
        for i in range(num_splits):
            output_dict = build_aggregated_model(test_site_num=i, split_type=split_type, split_num=i, pca=pca,
                                                 combine_all_local_tests=combine_all_local_tests)
            output_dict["split_type"] = split_type
            outputs.append(output_dict)

    return paput.pp_metrics(*outputs, columns=outputs[0].keys(), out_dir=output_path, out_filename=output_filename)


def cross_validate_distributed_model(split_types, num_splits, pca=False, combine_all_local_tests=True):
    output_filename = "FNC_distributed_cv" + ("_sep_test.csv" if combine_all_local_tests else ".csv")
    outputs = []
    local_outputs = []
    owner_node = 0
    columns = None

    for split_type in split_types:
        for i in range(num_splits):
            output_dict, local_output_dicts = build_distributed_model(owner_node_num=owner_node, split_type=split_type,
                                                                      split_num=i, pca=pca,
                                                                      combine_all_local_tests=combine_all_local_tests)
            # Add split_type
            output_dict["split_type"] = split_type
            local_output_dicts = [dict(local_dict, split_type=split_type) for local_dict in local_output_dicts]

            # Add split id
            output_dict["run_id"] = i
            local_output_dicts = [dict(local_dict, run_id=i) for local_dict in local_output_dicts]

            # Add node reference
            output_dict["node_id"] = "distributed"
            local_output_dicts = [dict(local_dict, node_id="local_" + str(indx + 1)) for indx, local_dict in
                                  enumerate(local_output_dicts)]

            columns = output_dict.keys() if columns is None else columns

            outputs.append(output_dict)
            local_outputs.append(local_output_dicts)

    owner_df = paput.pp_metrics(*outputs, columns=outputs[0].keys(), out_dir=output_path, out_filename=output_filename)
    local_df = paput.pp_metrics(*local_outputs, columns=outputs[0].keys(), out_dir=output_path,
                                out_filename="locals_" + output_filename)

    return owner_df, local_df

def get_latex_table(split_types, agg_df, dist_df):
    mean_agg_df = agg_df.groupby('split_type').mean().round(3)
    std_agg_df = agg_df.groupby('split_type').std().round(3)
    mean_dist_df = dist_df.groupby('split_type').mean().round(3)
    std_dist_df = dist_df.groupby('split_type').std().round(3)

    row_names = list(mean_dist_df.index)
    columns = list(mean_dist_df.columns)

    row_disp_names = {'random': 'random', 'age_stratified': 'age stratified',
                      'age_range_stratified': 'age-bin stratified'}

    print("Start latex code: \n")
    for row_name in split_types:
        print("&\\multirow{2}{*}{" + row_disp_names[row_name] + "} & Decentralized " +
              f"& ${mean_dist_df.loc[row_name]['rmse_train_aggregated']} \pm {std_dist_df.loc[row_name]['rmse_train_aggregated']}$"
              f"& ${mean_dist_df.loc[row_name]['rmse_test_aggregated']} \pm {std_dist_df.loc[row_name]['rmse_test_aggregated']}$"
              f"& ${mean_dist_df.loc[row_name]['mae_train_aggregated']} \pm {std_dist_df.loc[row_name]['mae_train_aggregated']}$"
              f"& ${mean_dist_df.loc[row_name]['mae_test_aggregated']} \pm {std_dist_df.loc[row_name]['mae_test_aggregated']}$"
              f"\\\\")

        print(f"& & Centralized "
              f"& ${mean_agg_df.loc[row_name]['rmse_train_aggregated']} \pm {std_agg_df.loc[row_name]['rmse_train_aggregated']}$"
              f"& ${mean_agg_df.loc[row_name]['rmse_test_aggregated']} \pm {std_agg_df.loc[row_name]['rmse_test_aggregated']}$"
              f"& ${mean_agg_df.loc[row_name]['mae_train_aggregated']} \pm {std_agg_df.loc[row_name]['mae_train_aggregated']}$"
              f"& ${mean_agg_df.loc[row_name]['mae_test_aggregated']} \pm {std_agg_df.loc[row_name]['mae_test_aggregated']}$"
              f"\\\\")
    print("\nDone latex code")



def stat_analysis(split_types, agg_df, dist_df):
    from scipy.stats import ttest_ind
    from scipy.stats import wilcoxon

    alpha = 0.05

    out_t_test = []
    out_wilcxn_test = []

    columns = ['rmse_train_aggregated', 'rmse_test_aggregated', 'mae_train_aggregated', 'mae_test_aggregated']

    for split_type in split_types:
        t_res = []
        wilcoxon_res = []
        for metric in columns:
            # print("Performing t-test for split_type: ", split_type, " metric: ", metric)

            agg_vals = agg_df[agg_df['split_type'] == split_type][metric]
            dist_vals = dist_df[dist_df['split_type'] == split_type][metric]

            stat, p = ttest_ind(agg_vals.values, dist_vals.values, equal_var=False)

            # print('stat=%.3f, p=%.3f' % (stat, p))
            res = 'Same' if p > alpha else 'Diff'
            t_res.append(res)

            # Non-parametric
            # print("Performing wilcoxon test for split_type: ", split_type, " metric: ", metric)
            stat, p = wilcoxon(agg_vals.values, dist_vals.values)
            # print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret

            res = 'Same' if p > alpha else 'Diff'
            wilcoxon_res.append(res)
            """
            if p > alpha:
                print('Same distribution (fail to reject H0)')
            else:
                print('Different distribution (reject H0)')
            """
        # Add results
        out_t_test.append(t_res)
        out_wilcxn_test.append(wilcoxon_res)

    print("t-test: \n", out_t_test)
    print("wilcoxon-test: \n", out_wilcxn_test)


def generate_cv_metrics(num_splits):
    split_types = ["random", "age_stratified", "age_range_stratified"]
    agg_df = cross_validate_aggregated_model(split_types=split_types, num_splits=num_splits, pca=False,
                                             combine_all_local_tests=True)
    dist_owner_df, dist_local_df = cross_validate_distributed_model(split_types=split_types, num_splits=num_splits,
                                                                    pca=False,
                                                                    combine_all_local_tests=True)

    stat_analysis(split_types, agg_df, dist_owner_df)

    # print latex table code
    get_latex_table(split_types, agg_df, dist_owner_df)

    # Generate box plot of the results of centralized and decentralized results
    pap_plt.plot_centralized_vs_decentralized(agg_df.copy(), dist_owner_df.copy(), output_path + "/FNC_box_")

    # Generate box plot comparision of decentralized local and owner results
    pap_plt.plot_local_vs_owner(dist_owner_df, dist_local_df, output_path + "/FNC_local_vs_dist" + ".png")


if __name__ == "__main__":
    """
    owner_node_num=0
    split_type = "age_stratified"
    print("owner_node: local"+str(owner_node_num))
    build_aggregated_model(owner_node_num,pca=False, split_type=split_type)
    """

    output_path = "/Users/sbasodi1/MEGA/work/trendz_2020/projects/brainage/brainage_fnc_rslt/paper/trends_decen_brainage/results"

    # build_aggregated_model(test_site_num=0, split_type=split_type, pca=False)

    # TODO: Make sure you generate partitions using "paper_dataset_generator.py" and run this script
    generate_cv_metrics(num_splits=5)
