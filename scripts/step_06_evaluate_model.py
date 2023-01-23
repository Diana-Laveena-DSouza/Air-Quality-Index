import argparse
import os
import logging
from utils.common import read_yaml, create_directories, save_json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import numpy as np
import pickle as pkl

STAGE = "MODEL EVALUATION"

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    #Create Directories
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    x_test_file = artifacts["X_TEST"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    create_directories([split_data_dir_path])
    x_test_data_path = os.path.join(split_data_dir_path, x_test_file)
    model_dir = artifacts["MODEL_DIR"]
    model_file = artifacts["MODEL_NAME"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])
    model_file_path = os.path.join(model_dir_path, model_file)

    #Load Files
    x_test = pd.read_csv(x_test_data_path)
    x_test_aqi = x_test.drop(['AQI'], axis=1)
    y_test_aqi = x_test['AQI']

    #Model Evaluation
    model = pkl.load(open(model_file_path, "rb"))
    pred_test = model.predict(x_test_aqi)
    r2sore = r2_score(y_test_aqi, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_aqi, pred_test))
    rmsle = np.sqrt(mean_squared_log_error(y_test_aqi, pred_test))
    model_name = model.__class__.__name__

   # Scores
    scores = {"MODEL_NAME": model_name, "R2": r2sore, "RMSE": rmse, "RMSLE": rmsle}
    scores_file_path = config["scores"]
    save_json(scores_file_path, scores)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path = parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e