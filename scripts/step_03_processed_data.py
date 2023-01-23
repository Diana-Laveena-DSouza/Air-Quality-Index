import pandas as pd
import numpy as np
import argparse
import os
import logging
from utils.common import read_yaml, create_directories
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


STAGE = "PROCESS DATA"

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
)


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    get_local_dir = artifacts["GET_DATA_DIR"]
    get_local_file = artifacts["GET_DATA_FILE"]
    get_local_dir_path = os.path.join(artifacts_dir, get_local_dir)
    create_directories([get_local_dir_path])
    get_local_filepath = os.path.join(get_local_dir_path, get_local_file)
    process_local_dir = artifacts["PROCESS_LOCAL_DIR"]
    process_local_file = artifacts["PROCESS_DATA"]
    outliers_after_handling = artifacts["OUTLIERS_AFTER_HANDLING"]
    process_local_dir_path = os.path.join(artifacts_dir, process_local_dir)
    create_directories([process_local_dir_path])
    process_local_file_path = os.path.join(process_local_dir_path, process_local_file)
    outliers_after_handling_path = os.path.join(process_local_dir_path, outliers_after_handling)
    model_dir = artifacts["MODEL_DIR"]
    model_file = artifacts["ONE_HOT_MODEL"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])
    model_file_path = os.path.join(model_dir_path, model_file)

    #Load File
    data = pd.read_csv(get_local_filepath)
    logging.info(data.head())
    # Remove rows having more than 30% missing values
    data_new_no_row_miss = data[data.isna().sum(axis=1)/len(data.columns)<=0.3]
    logging.info(f"After removing rows having more than 30% missing values: \n {data_new_no_row_miss.head()}")
    logging.info(f"Number of records in new data is saved at: {len(data_new_no_row_miss)}")
    logging.info(data_new_no_row_miss.head())

    # Filling the Missing values
    data_impute = data_new_no_row_miss.bfill(axis = 'rows').ffill(axis = 'rows')

    # Treating Outliers
    columns = list(set(data_impute.columns) - {'AQI_Bucket', 'State', 'City', 'Station', 'From Date', 'To Date', 'AQI'})
    data_rem_out = np.cbrt(data_impute[columns])
    data_rem_out['AQI_Bucket'] = data_impute['AQI_Bucket']
    data_rem_out['City'] = data_impute['City']
    data_rem_out['State'] = data_impute['State']
    data_rem_out['Station'] = data_impute['Station']
    data_rem_out['From Date'] = data_impute['From Date']
    data_rem_out['To Date'] = data_impute['To Date']
    data_rem_out['AQI'] = data_impute['AQI']

    logging.info(f"checking skewness of columns after handling outliers\n")
    columns = list(
        set(data_rem_out.columns) - {'AQI_Bucket', 'State', 'City', 'Station', 'From Date', 'To Date', 'AQI'})
    for col in columns:
        logging.info(f"{col}, ' : ', {data_rem_out[col].skew()}")

    # checking outliers after handling it
    columns = list(set(data_rem_out.columns) - {'AQI_Bucket', 'State', 'City', 'Station', 'From Date', 'To Date', 'AQI'})
    q, r = divmod(len(columns), 4)
    fig, ax = plt.subplots(q + 1, 4, figsize = (20, 20))
    for i, col in enumerate(columns):
        q, r = divmod(i, 4)
        sbn.boxplot(data = data_rem_out, x = col, ax = ax[q, r])
    plt.savefig(outliers_after_handling_path)

    # LabelEncoding
    LEncoder = LabelEncoder().fit(data_rem_out['AQI_Bucket'])
    data_rem_out['AQI_Bucket'] = LEncoder.transform(data_rem_out['AQI_Bucket'])

    # One Hot Encoding
    one_hot = OneHotEncoder(drop='first').fit(data_rem_out[['City', 'State', 'Station']])
    data_cat = one_hot.transform(data_rem_out[['City', 'State', 'Station']]).toarray()
    data_cat_final = pd.DataFrame(data_cat)
    data_cat_final.columns = one_hot.get_feature_names_out(['City', 'State', 'Station'])


    # Saving the model
    pkl.dump(one_hot, open(model_file_path, 'wb'))
    data_onhot = pd.concat([data_rem_out, data_cat_final], axis=1)
    data_onhot['To Date'] = pd.to_datetime(data_onhot['To Date'], format='%d-%m-%Y %H:%M')
    data_onhot['Year'] = data_onhot['To Date'].dt.year
    data_onhot['Month'] = data_onhot['To Date'].dt.month
    data_onhot['Day'] = data_onhot['To Date'].dt.day

    #Saving the File
    data_onhot.to_csv(process_local_file_path, index = False)

    logging.info(f"processed data is saved at: {process_local_file_path}")
    logging.info(f"One Hot Encoding Model is saved at: {model_file_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path = parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e