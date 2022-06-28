import pickle
import mlflow
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import argparse
import logging

logger = logging.getLogger("score")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("score.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_flag = 1

if console_flag == 1:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
else:
    pass

default_op_path = os.getcwd() + "\\artifacts"

parser = argparse.ArgumentParser()
parser.add_argument(
    "path_1", help="Path to the previously saved model pickle file with file name and extension", nargs="?", const=default_op_path + "\\model.pickle"
)
parser.add_argument("path_2", help="Path to saved output files", nargs="?", const=default_op_path)
args = parser.parse_args()
mpath = args.path_1
fpath = args.path_2

if mpath is None:
    mpath = default_op_path + "\\model.pickle"
if fpath is None:
    fpath = default_op_path

# mpath = "C:\\Users\\saikumar.godha\\Downloads\\basic_regg_lib\\output_folder\\model.pickle" #Pickle path
# fpath = "C:\\Users\\saikumar.godha\\Downloads\\basic_regg_lib\\output_folder" #csv files folder path

X_test = pd.read_csv(os.path.join(fpath, "X_test.csv"))
y_test = pd.read_csv(os.path.join(fpath, "y_test.csv"))

remote_server_uri = "http://127.0.0.1:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env


def validation(X_test, y_test, mpath):
    with mlflow.start_run(run_name="score") as parent_run:
        mlflow.log_param("parent", "yes")
        # load the model from disk
        loaded_model = pickle.load(open(mpath, "rb"))
        y_pred = loaded_model.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        VarScore = metrics.explained_variance_score(y_test, y_pred)
        r_square = metrics.r2_score(y_test, y_pred)
        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "VarScore": VarScore, "r2": r_square})

        logger.debug({"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "VarScore": VarScore, "r2": r_square})
    return 1


validation(X_test, y_test, mpath)
