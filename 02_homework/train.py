import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow 
from mlflow.tracking import MlflowClient



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)



def run_train(data_path):
    mlflow.set_experiment("homework-1")
    # MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        # mlflow.log_artifact(local_path="../models/rf.bin", artifact_path="models_pickle")

if __name__ == '__main__':

    run_train()
