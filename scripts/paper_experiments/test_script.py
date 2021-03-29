
from pooled import aggregated_SVR
from svr_utils import form_XYMatrices


def test_local_model(client_name):
    input_dir_name = "../test/input/"+client_name+"/simulatorRun/age_range_stratified/"
    train_file_name=client_name+"_fnc_age_train.csv"
    test_file_name=client_name+"_fnc_age_test.csv"
    X_train,y_train = form_XYMatrices(input_dir_name, train_file_name)
    X_test,y_test = form_XYMatrices(input_dir_name, test_file_name)

    aggregated_SVR(X_train, X_test, y_train, y_test)
    print("Done")


if __name__ == "__main__":
    test_local_model("local0")

    for i in range(5):
        client_name="local"+str(i+1)
        print("Training model for data in: "+ client_name )
        test_local_model(client_name)
        print("Done with"+client_name+"\n")
