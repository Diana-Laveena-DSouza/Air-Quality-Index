import argparse
import os
import logging
from utils.common import read_yaml, create_directories
import pandas as pd
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle as pkl
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

STAGE = "MODEL TRAINING"

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
    model_dir = artifacts["MODEL_DIR"]
    model_file = artifacts["MODEL_NAME"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])
    model_file_path = os.path.join(model_dir_path, model_file)
    regressor_scores_dir = artifacts["OVER_REG_SCORES_DIR"]
    classifier_before = artifacts["CLASS_BEFORE"]
    classifier_after = artifacts["CLASS_AFTER"]
    regressor_before = artifacts["REG_BEFORE"]
    regressor_after = artifacts["REG_AFTER"]
    regressor_scores_dir_path = os.path.join(artifacts_dir, regressor_scores_dir)
    create_directories([regressor_scores_dir_path])
    classifier_before_path = os.path.join(regressor_scores_dir_path, classifier_before)
    classifier_after_path = os.path.join(regressor_scores_dir_path, classifier_after)
    regressor_before_path = os.path.join(regressor_scores_dir_path, regressor_before)
    regressor_after_path = os.path.join(regressor_scores_dir_path, regressor_after)

    # Parameters
    parameters = params["model_params"]
    n_estimators = parameters["n_estimators"]
    max_depth = parameters["max_depth"]
    min_samples_split = parameters["min_samples_split"]
    max_features = parameters["max_features"]

    # Load Files
    x_train = pd.read_csv(x_train_data_path)
    y_train = pd.read_csv(y_train_data_path)
    x_test = pd.read_csv(x_test_data_path)
    y_test = pd.read_csv(y_test_data_path)
    x_train_over = pd.read_csv(x_train_over_data_path)
    y_train_over = pd.read_csv(y_train_over_data_path)

    # Model Training and Selection
    # 1. Check the Model accuracy for classification to check the oversampling effect
    # Before Oversampling Technique
    model_factory = [SVC(), XGBClassifier(), DecisionTreeClassifier(), AdaBoostClassifier(), BaggingClassifier(),
                     RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier()]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    model_name = []
    for model in model_factory:
        print(model.__class__.__name__)

        model_train = model.fit(x_train, y_train)
        pred_test = model_train.predict(x_test)
        accuracy.append(accuracy_score(y_test, pred_test))
        precision.append(precision_score(y_test, pred_test,
                                         average='macro'))
        recall.append(recall_score(y_test, pred_test,
                                   average='macro'))
        f1.append(f1_score(y_test, pred_test,
                           average='macro'))
        model_name.append(model.__class__.__name__)
    classifier_before_oversample = pd.concat(
        [pd.Series(model_name), pd.Series(accuracy), pd.Series(precision), pd.Series(recall), pd.Series(f1)], axis=1)
    classifier_before_oversample.columns = ["Model Name", "Accuracy", "Precision", "Recall", "F1-Score"]
    classifier_before_oversample.to_csv(classifier_before_path)

    # After Oversampling Technique
    model_factory = [SVC(), XGBClassifier(), DecisionTreeClassifier(), AdaBoostClassifier(), BaggingClassifier(),
                     RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier()]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    model_name = []
    for model in model_factory:
        print(model.__class__.__name__)
        model_train = model.fit(x_train_over, y_train_over)
        pred_test = model_train.predict(x_test)
        accuracy.append(accuracy_score(y_test, pred_test))
        precision.append(precision_score(y_test, pred_test, average='macro'))
        recall.append(recall_score(y_test, pred_test, average='macro'))
        f1.append(f1_score(y_test, pred_test, average='macro'))
        model_name.append(model.__class__.__name__)
    classifier_after_oversample = pd.concat(
        [pd.Series(model_name), pd.Series(accuracy), pd.Series(precision), pd.Series(recall), pd.Series(f1)], axis=1)
    classifier_after_oversample.columns = ["Model Name", "Accuracy", "Precision", "Recall", "F1-Score"]
    classifier_after_oversample.to_csv(classifier_after_path)

    # 2. Check the Model accuracy for Regressor to check the oversampling effect
    # Before Over Sampling
    x_train_aqi = x_train.drop(['AQI'], axis=1)
    y_train_aqi = x_train['AQI']
    x_test_aqi = x_test.drop(['AQI'], axis=1)
    y_test_aqi = x_test['AQI']

    model_factory = [SVR(), XGBRegressor(), DecisionTreeRegressor(), AdaBoostRegressor(), BaggingRegressor(),
                     RandomForestRegressor(), GradientBoostingRegressor(), KNeighborsRegressor()]
    r2sore = []
    rmse = []
    rmsle = []
    model_name = []
    for model in model_factory:
        model_train = model.fit(x_train_aqi, y_train_aqi)
        pred_test = model_train.predict(x_test_aqi)
        r2sore.append(r2_score(y_test_aqi, pred_test))
        rmse.append(np.sqrt(mean_squared_error(y_test_aqi, pred_test)))
        rmsle.append(np.sqrt(mean_squared_log_error(y_test_aqi, pred_test)))
        model_name.append(model.__class__.__name__)
    regressor_before_oversample = pd.concat(
        [pd.Series(model_name), pd.Series(r2sore), pd.Series(rmse), pd.Series(rmsle)], axis=1)
    regressor_before_oversample.columns = ["Model Name", "R2", "RMSE", "RMSLE"]
    regressor_before_oversample.to_csv(regressor_before_path)

    # After Over Sampling
    x_train_aqi = x_train_over.drop(['AQI'], axis=1)
    y_train_aqi = x_train_over['AQI']
    x_test_aqi = x_test.drop(['AQI'], axis=1)
    y_test_aqi = x_test['AQI']

    model_factory = [SVR(), XGBRegressor(), DecisionTreeRegressor(), AdaBoostRegressor(), BaggingRegressor(),
                     RandomForestRegressor(), GradientBoostingRegressor(), KNeighborsRegressor()]
    r2sore = []
    rmse = []
    rmsle = []
    model_name = []
    for model in model_factory:
        model_train = model.fit(x_train_aqi, y_train_aqi)
        pred_test = model_train.predict(x_test_aqi)
        r2sore.append(r2_score(y_test_aqi, pred_test))
        rmse.append(np.sqrt(mean_squared_error(y_test_aqi, pred_test)))
        rmsle.append(np.sqrt(mean_squared_log_error(y_test_aqi, pred_test)))
        model_name.append(model.__class__.__name__)
    regressor_after_oversample = pd.concat(
        [pd.Series(model_name), pd.Series(r2sore), pd.Series(rmse), pd.Series(rmsle)], axis=1)
    regressor_after_oversample.columns = ["Model Name", "R2", "RMSE", "RMSLE"]
    regressor_after_oversample.to_csv(regressor_after_path)

    # Selected Model is RandomForest gives highest accuracy on oversampling data
    model_train = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth,
                                        min_samples_split = min_samples_split, max_features = max_features).fit(x_train_aqi, y_train_aqi)
    pkl.dump(model_train, open(model_file_path, "wb"))

    logging.info(f"Results before oversampling are saved at: {classifier_before_path}, {regressor_before_path}")
    logging.info(f"Results after oversampling are saved at: {classifier_after_path}, {regressor_after_path}")
    logging.info(f"Model is saved at: {model_file_path}")

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