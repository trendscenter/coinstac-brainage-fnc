import os
import h5py

import numpy as np
import pandas as pd

from random import shuffle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

COL_NAME_SUBJECT_ID = "eid"
COL_NAME_AGE = "age_when_attended_assessment_centre_f21003_2_0"

def get_eid_age_from_file(file_dir, file_name):
    df = pd.read_table(os.path.join(file_dir, file_name))
    df_eid_age = df[[COL_NAME_SUBJECT_ID, COL_NAME_AGE]]
    #Need to perform inner join with eid of this df with filename of fN variable in mat file

    return df_eid_age


def get_eid_from_fN(fN_data):
    eid_list=[]
    for data in fN_data:
        eid_list.append(int(data.split("/")[8].split("_")[0]))

    df_eid=pd.DataFrame(eid_list,columns =[COL_NAME_SUBJECT_ID])

    return df_eid


def read_fnc(file_dir, file_name):
    def get_cell_array_data(key_name):
        data = []
        for column in f[key_name]:
            row_data = []
            for row_number in range(len(column)):
                #eid
                row_data.append(''.join(map(chr, f[column[row_number]][:])))
            data.append(row_data)
        return data


    def get_numpy_array(key_name):
        temp=np.array(f[key_name])
        return np.transpose(temp, tuple(range(len(temp.shape)-1,-1,-1)))


    with h5py.File(os.path.join(file_dir, file_name), 'r') as f:
        #for k, v in f.items():
        #    mat_file_dict[k] = np.array(v)
        fN_data=get_cell_array_data('fN')[0]
        icn_ins_data=get_numpy_array('icn_ins')
        fnc_data=get_numpy_array('corrdata')

    return fN_data, fnc_data, icn_ins_data


def generate_XYMatrices(input_dir, data_file, label_file):
    fN_data, fnc_data, icn_ins_data = read_fnc(input_dir, data_file)
    df_eid = get_eid_from_fN(fN_data)

    df_eid_age = get_eid_age_from_file(input_dir, label_file)

    df_result_eid = pd.concat([df_eid_age, df_eid], axis=1, join="inner").reindex(df_eid.index)
    y = df_result_eid[[COL_NAME_AGE]].to_numpy()
    #y=y.reshape((y.shape[0]))

    #X = fnc_data.reshape((fnc_data.shape[0],-1))
    # Extract only upper triangular data
    upper_tri_indx=np.triu_indices(fnc_data.shape[1], k = 1)
    X=np.empty((len(fnc_data), len(upper_tri_indx[0])))
    for i in range(len(fnc_data)):
        X[i] = fnc_data[i][upper_tri_indx]

    return (X, y, fN_data)



def get_age_range_stratified_splits(k, X, y, shuffle_data, test_size, num_bins=8):
    k_splits=[]
    # Divide ages into equal size bins and label them
    y_age_range=pd.qcut(y.reshape(y.shape[0]), num_bins, labels=False)

    kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=42)
    for tr_indx, tt_indx in kfold.split(X, y_age_range):
        shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=test_size)
        indx_split=list(shufflesplit.split(X[tt_indx], y_age_range[tt_indx]))[0]
        train_index= tt_indx[indx_split[0]]
        test_index= tt_indx[indx_split[1]]

        k_splits.append([train_index, test_index])

    return k_splits

def get_age_stratified_splits(k, X, y, shuffle_data, test_size):
    k_splits=[]

    kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=42)
    for tr_indx, tt_indx in kfold.split(X, y):
        """
        Note: Using random split instead of Stratified split. Stratified split does not work as the data in y are
         real values and based on the contents in y it sometimes causes 
        the following error:  ValueError("The least populated class in y has only 1....")
        """
        train_index, test_index = train_test_split(tt_indx, shuffle=True, test_size=test_size)
        k_splits.append([train_index, test_index])

    return k_splits



def get_random_splits(k, X, y, shuffle_data, test_size):
    k_splits=[]
    indx_arr=np.arange(len(X))

    if shuffle_data:
        shuffle(indx_arr)

    data_k_splits = np.array_split(indx_arr, k)
    for i in range(len(data_k_splits)):
        spt_indx=data_k_splits[i]
        train_index, test_index = train_test_split(spt_indx, shuffle=True, test_size=test_size)
        k_splits.append([train_index, test_index])

    return k_splits


def generate_k_splits(X, y, fN_data, save_to_dir, k=6, shuffle_data=True, type="age_stratified", test_size=0.1):

    assert len(X) == len(y) == len(fN_data)

    #concatenate
    combined_xy= np.concatenate((X, y),axis = 1)
    fN_data_arr = np.array(fN_data)


    if type== "random":
        k_splits = get_random_splits(k, X, y, shuffle_data, test_size)
    elif type=="age_stratified":
        k_splits = get_age_stratified_splits(k, X, y, shuffle_data, test_size)
    elif type=="age_range_stratified":
        k_splits = get_age_range_stratified_splits(k, X, y, shuffle_data, test_size)
    else :
        raise Exception ("Please specify the type of split.")

    for i in range(len(k_splits)):

        train_index = k_splits[i][0]
        test_index = k_splits[i][1]

        local_dir=save_to_dir + os.sep + f"local{i}" + os.sep + "simulatorRun" +os.sep +type
        os.makedirs(local_dir, exist_ok=True)
        np.savetxt(local_dir + os.sep + f"local{i}_fnc_age_train.csv", combined_xy[train_index], delimiter=",")
        np.savetxt(local_dir + os.sep + f"local{i}_fnc_age_test.csv", combined_xy[test_index], delimiter=",")
        np.savetxt(local_dir + os.sep + f"local{i}_subject_ref_filename_train.csv", fN_data_arr[train_index],  fmt='%s', delimiter=",")
        np.savetxt(local_dir + os.sep + f"local{i}_subject_ref_filename_test.csv", fN_data_arr[test_index],  fmt='%s', delimiter=",")




def generate_k_random_splits(X, y, fN_data, save_to_dir, k=6, shuffle_data=True):

    assert len(X) == len(y) == len(fN_data)

    #concatenate
    combined_xy= np.concatenate((X, y),axis = 1)
    fN_data_arr = np.array(fN_data)
    indx_arr=np.arange(len(X))

    if shuffle_data:
        shuffle(indx_arr)

    indx_splits = np.array_split(indx_arr, k)
    for i in range(len(indx_splits)):
        split_xy= combined_xy[indx_splits[i]]
        split_fN= fN_data_arr[indx_splits[i]]

        local_dir=save_to_dir + os.sep + f"local{i}" + os.sep + "simulatorRun"
        os.makedirs(local_dir, exist_ok=True)
        np.savetxt(local_dir + os.sep + f"local{i}_fnc_age.csv", split_xy, delimiter=",")
        np.savetxt(local_dir + os.sep + f"local{i}_subject_ref_filename.csv", split_fN,  fmt='%s', delimiter=",")

def generate_on_server():
    dir_name=""
    age_dir="/data/mialab/competition2019/UKBiobank/packNship/results/scores/new/"
    age_file_name="ukb_unaffected_clean_select_log_with_sex.tab"

    fnc_dir="/data/mialab/competition2019/UKBiobank/FNC3/results/non-mc/"
    fnc_file_name="data_staticFC_11754.mat"

    #[X, y, fN_data]= generate_XYMatrices(dir_name, data_file=fnc_file_name, label_file=age_file_name)
    [X, y, fN_data]= generate_XYMatrices(dir_name, data_file=fnc_dir+fnc_file_name, label_file=age_dir+age_file_name)
    #generate_k_splits(X, y, fN_data, save_to_dir="../test/input", k=6, shuffle_data=True)
    #aggregated_SVR(X, y)

def generate_on_desktop():
    dir_name= "/Users/sbasodi1/workplace/brain_age_pipeline/fnc/UKBioBank_Comp2019/FNC3/results/non-mc/"

    age_file_name="ukb_unaffected_clean_select_log_with_sex.tab"

    fnc_file_name="data_staticFC_11754.mat"

    [X, y, fN_data]= generate_XYMatrices(dir_name, data_file=fnc_file_name, label_file=age_file_name)
    generate_k_splits(X, y, fN_data, save_to_dir="../test/input", k=6, shuffle_data=True, type="age_range_stratified", test_size=0.1)
    #aggregated_SVR(X, y)

def test_functions():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4],
                  [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([0, 1, 2, 3, 1, 1, 2, 3, 2 ,0, 1, 3, 0,3, 2 ,0, 1, 3, 0])
    y= np.array(range(len(X)))

    #get_random_splits(3, X, y, shuffle_data=True, test_size=0.1)
    #generate_k_splits(X, y, y, "", k=3, shuffle_data=True, type="age_range_stratified")
    get_age_range_stratified_splits(2, X, y, shuffle_data=True, test_size=0.5, num_bins=4)

if __name__ == "__main__":
    #generate_on_server()
    generate_on_desktop()
    #test_functions()

