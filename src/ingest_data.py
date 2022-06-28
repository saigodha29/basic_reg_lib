import os
import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from traitlets import default
import logging

logger = logging.getLogger("ingest_data")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("ingest_data.log")
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
# fpath = "C:\\Users\\saikumar.godha\\Downloads\\basic_regg_lib\\artifacts"

data_path = os.getcwd() + "\\src\\house_data.csv"
Data = pd.read_csv(data_path)

parser = argparse.ArgumentParser()
parser.add_argument("path_1", help="Path to save output files", nargs="?", const=default_op_path)
args = parser.parse_args()
fpath = args.path_1


if fpath is None:
    fpath = default_op_path

# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
remote_server_uri = "http://127.0.0.1:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env


def data_prep(Data, fpath):
    with mlflow.start_run(run_name="Data Prep") as parent_run:
        mlflow.log_param("parent", "yes")
        Data = Data.drop("date", axis=1)
        Data = Data.drop("id", axis=1)
        Data = Data.drop("zipcode", axis=1)

        X = Data.drop("price", axis=1).values
        y = Data["price"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        try:
            os.makedirs(fpath)
        except FileExistsError:
            # directory already exists
            pass

        pd.DataFrame(X_train).to_csv(os.path.join(fpath, "X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(fpath, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(fpath, "y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(fpath, "y_test.csv"), index=False)

        logger.debug("files exported")
    return 1


data_prep(Data, fpath)
