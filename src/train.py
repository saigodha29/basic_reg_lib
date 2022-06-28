from sklearn.ensemble import RandomForestRegressor
import mlflow
import pickle
import pandas as pd
import os
import argparse
import logging

logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("train.log")
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
parser.add_argument("path_1", help="Path to saved output files", nargs="?", const=default_op_path)
args = parser.parse_args()
fpath = args.path_1


if fpath is None:
    fpath = default_op_path

# fpath = "C:\\Users\\saikumar.godha\\Downloads\\basic_regg_lib\\output_folder"

remote_server_uri = "http://127.0.0.1:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

X_train = pd.read_csv(os.path.join(fpath, "X_train.csv"))
y_train = pd.read_csv(os.path.join(fpath, "y_train.csv"))


def train_model(X_train, y_train, fpath):
    with mlflow.start_run(run_name="Model") as parent_run:
        mlflow.log_param("parent", "yes")
        m = RandomForestRegressor(n_jobs=-1, oob_score=True)
        m.fit(X_train, y_train)

        # save the model to disk
        filename = os.path.join(fpath, "model.pickle")
        pickle.dump(m, open(filename, "wb"))

        logger.debug("model saved")
    return 1


print(train_model(X_train, y_train, fpath))
