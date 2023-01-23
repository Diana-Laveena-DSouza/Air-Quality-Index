import argparse
import os
import logging
from utils.common import read_yaml, create_directories
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


STAGE = "TRAIN TEST SPLIT"

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

    # Load the Split Files
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    split_data_dir = artifacts["SPLIT_DATA_DIR"]
    x_train_file = artifacts["X_TRAIN"]
    x_test_file = artifacts["X_TEST"]
    y_train_file = artifacts["Y_TRAIN"]
    y_test_file = artifacts["Y_TEST"]
    x_train_over_file = artifacts["X_TRAIN_OVER"]
    y_train_over_file = artifacts["Y_TRAIN_OVER"]
    split_data_dir_path = os.path.join(artifacts_dir, split_data_dir)
    create_directories([split_data_dir_path])
    x_train_data_path = os.path.join(split_data_dir_path, x_train_file)
    x_test_data_path = os.path.join(split_data_dir_path, x_test_file)
    y_train_data_path = os.path.join(split_data_dir_path, y_train_file)
    y_test_data_path = os.path.join(split_data_dir_path, y_test_file)
    x_train_over_data_path = os.path.join(split_data_dir_path, x_train_over_file)
    y_train_over_data_path = os.path.join(split_data_dir_path, y_train_over_file)

    process_local_dir = artifacts["PROCESS_LOCAL_DIR"]
    process_local_file = artifacts["PROCESS_DATA"]
    process_local_dir_path = os.path.join(artifacts_dir, process_local_dir)
    create_directories([process_local_dir_path])
    process_local_file_path = os.path.join(process_local_dir_path, process_local_file)

    # Parameters
    train_test_split_ = params["train_test_split"]
    test_size = train_test_split_["TEST_SIZE"]
    random_state = train_test_split_["RANDOM_STATE"]
    over_sample = params["over_sample"]
    random_state_over = over_sample["RANDOM_STATE_OVER"]

    # Load File
    data = pd.read_csv(process_local_file_path)

    #Feature and Target for Classification
    Feature_class = data.drop(['AQI_Bucket', 'State', 'City', 'Station', 'From Date', 'To Date'], axis = 1)
    Target_class = data['AQI_Bucket']

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(Feature_class, Target_class, test_size = test_size,
                                                        random_state = random_state)
    logging.info(f"x_train_shape: {x_train.shape}, x_test_shape: {x_test.shape}, y_train_shape: {y_train.shape}, y_test_shape: {y_test.shape}")

    # OverSampling

    ros = RandomOverSampler(random_state = random_state_over)
    x_train_over, y_train_over = ros.fit_resample(x_train, y_train)
    logging.info(f"x_train_over_shape: {x_train_over.shape}, y_train_over_shape: {y_train_over.shape}")

    # Save the Files
    x_train.to_csv(x_train_data_path, index = False)
    y_train.to_csv(y_train_data_path, index = False)
    x_test.to_csv(x_test_data_path, index = False)
    y_test.to_csv(y_test_data_path, index = False)
    x_train_over.to_csv(x_train_over_data_path, index=False)
    y_train_over.to_csv(y_train_over_data_path, index=False)

    logging.info(f"Train Test Split Files are saved at: {x_train_data_path}, {y_train_data_path}, {x_test_data_path}, {y_test_data_path}")
    logging.info(f"Oversampling Files are saved at: {x_train_over_data_path}, {y_train_over_data_path}")

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