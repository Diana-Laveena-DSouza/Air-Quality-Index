import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import argparse
import os
import logging
from utils.common import read_yaml, create_directories

STAGE = "EDA ANALYSIS"

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

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    get_local_dir = artifacts["GET_DATA_DIR"]
    get_local_file = artifacts["GET_DATA_FILE"]
    get_local_dir_path = os.path.join(artifacts_dir, get_local_dir)
    create_directories([get_local_dir_path])
    get_local_filepath = os.path.join(get_local_dir_path, get_local_file)
    eda_data_dir = artifacts["EDA_DIR"]
    miss_col_file = artifacts["MISS_COL"]
    miss_row_file = artifacts["MISS_ROW"]
    dist_stat = artifacts["DIS_STAT"]
    dist_image = artifacts["DIST_IMAGE"]
    outliers = artifacts["CHECK_OUTLIERS"]
    correlation = artifacts["CORR_IMAGE"]
    pollutant_period = artifacts["POLL_PERIOD"]
    count_aqib = artifacts["COUNT_AQIB"]
    min_conc = artifacts["MIN_CONC"]
    max_conc = artifacts["MAX_CONC"]
    mean_conc = artifacts["MEAN_CONC"]
    pollutant_impact = artifacts["POLLUTANT_IMPACT"]
    aqi_impact = artifacts["AQI_IMPACT"]
    eda_data_dir_path = os.path.join(artifacts_dir, eda_data_dir)
    create_directories([eda_data_dir_path])
    miss_col_file_path = os.path.join(eda_data_dir_path, miss_col_file)
    miss_row_file_path = os.path.join(eda_data_dir_path, miss_row_file)
    dist_stat_path = os.path.join(eda_data_dir_path, dist_stat)
    dist_image_path = os.path.join(eda_data_dir_path, dist_image)
    outliers_path = os.path.join(eda_data_dir_path, outliers)
    correlation_path = os.path.join(eda_data_dir_path, correlation)
    pollutant_period_path = os.path.join(eda_data_dir_path, pollutant_period)
    count_aqib_path = os.path.join(eda_data_dir_path, count_aqib)
    min_conc_path = os.path.join(eda_data_dir_path, min_conc)
    max_conc_path = os.path.join(eda_data_dir_path, max_conc)
    mean_conc_path = os.path.join(eda_data_dir_path, mean_conc)
    pollutant_impact_path = os.path.join(eda_data_dir_path, pollutant_impact)
    aqi_impact_path = os.path.join(eda_data_dir_path, aqi_impact)

    #Load file
    data = pd.read_csv(get_local_filepath)

    # data types
    logging.info(f"Data Types:\n {data.dtypes}")

    # shape of the data
    logging.info(f"Shape of the Data:\n {data.shape}")

    # missing values in the columns in %
    missing_cols_v = pd.DataFrame(data.isna().sum() / len(data))
    missing_cols_v.columns = ["missing_columns_values"]
    missing_cols_v.to_csv(miss_col_file_path)

    # missing values in the rows in %
    missing_rows_v = pd.DataFrame(data.isna().sum(axis = 1) / len(data.columns))
    missing_rows_v.columns = ["missing_rows_values"]
    missing_rows_v.to_csv(miss_row_file_path)

    # descriptive statistics
    data_des = data.describe()
    data_des.to_csv(dist_stat_path)

    # data distribution
    columns = list(set(data.columns) - {'State', 'City', 'Station',	'From Date', 'To Date', 'AQI_Bucket'})
    q, r = divmod(len(columns), 4)
    fig, ax = plt.subplots(q, 4, figsize = (20, 20))
    for i, col in enumerate(columns):
        q, r = divmod(i, 4)
        sbn.histplot(data[col], ax = ax[q, r], kde = True)
    plt.savefig(dist_image_path)

    # checking outliers
    columns = list(set(data.columns) - {'State', 'City', 'Station',	'From Date', 'To Date', 'AQI_Bucket'})
    q, r = divmod(len(columns), 4)
    fig, ax = plt.subplots(q, 4, figsize = (20, 20))
    for i, col in enumerate(columns):
        q, r = divmod(i, 4)
        sbn.boxplot(data = data, x = col, ax = ax[q, r])
    plt.savefig(outliers_path)

    # checking skewness of columns
    columns = list(set(data.columns) - {'State', 'City', 'Station',	'From Date', 'To Date', 'AQI_Bucket'})
    for col in columns:
        logging.info(f"Checking the Skewness of Columns: {data[col].skew()}")

    # Correlation Analysis
    plt.figure(figsize = (20, 20))
    sbn.heatmap(data.corr(), annot = True)
    plt.savefig(correlation_path)

    # Checking concentration of pollutants over the period
    # Changing Date datatype
    data['To Date'] = pd.to_datetime(data['To Date'])
    columns = list(set(data.columns) - {'State', 'City', 'Station',	'From Date', 'AQI', 'To Date', 'AQI_Bucket'})
    q, r = divmod(len(columns), 1)
    fig, ax = plt.subplots(q, 1, figsize = (15, 40))
    for i, col in enumerate(columns):
        q, r = divmod(i, 1)
        sbn.lineplot(data = data, x = 'To Date', y = col, ax = ax[q])
    plt.savefig(pollutant_period_path)

    # Count of unique values in AQIBucket
    plt.figure(figsize = (10, 10))
    sbn.countplot(data = data, x = 'AQI_Bucket')
    plt.savefig(count_aqib_path)

    # Minimum concentration of each pollutant over the city
    columns = list(set(data.columns) - {'State', 'Station',	'From Date', 'To Date', 'AQI', 'AQI_Bucket'})
    data_min = data.loc[:, columns].groupby('City').min()
    data_min.to_csv(min_conc_path)

    # Maximum concentration of each pollutant over the city
    columns = list(set(data.columns) - {'State', 'Station',	'From Date', 'To Date', 'AQI', 'AQI_Bucket'})
    data_max = data.loc[:, columns].groupby('City').max()
    data_max.to_csv(max_conc_path)

    # Mean concentration of each pollutant over the city
    columns = list(set(data.columns) - {'State', 'Station',	'From Date', 'To Date', 'AQI', 'AQI_Bucket'})
    data_mean = data.loc[:, columns].groupby('City').mean()
    data_mean.to_csv(mean_conc_path)

    # Distribution of pollutants based on city
    columns = list(set(data.columns) - {'State', 'Station', 'City',	'From Date', 'To Date', 'AQI', 'AQI_Bucket'})
    for i, col in enumerate(columns):
        plt.figure(figsize = (10, 10))
        sbn.boxplot(data = data, x = col, y = 'City')
        plt.xticks(rotation = 90)
        plt.savefig(os.path.join(eda_data_dir_path, "city_" + col + ".jpg"))

        #pollutants impact on cities BVHB

        columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'Ozone', 'NH3']
        fig = plt.figure(figsize=(35, 35))
        for i, col in enumerate(columns):
            q, r = divmod(i, 4)
            ax = fig.add_subplot(int(str(q + 1) + "4" + str(r + 1)))
            data_mean = data_mean.fillna(0)
            ax.pie(data_mean[col], labels=data_mean.index, autopct='%.2f%%')
            plt.xlabel(col)
        plt.savefig(pollutant_impact_path)

        #AQI impact on cities over the years
        columns = ['Bengaluru', 'Lucknow', 'Patna', 'Delhi']
        q, r = divmod(len(columns), 1)
        fig, ax = plt.subplots(q, 1, figsize=(20, 40))
        for i, col in enumerate(columns):
            q, r = divmod(i, 1)
            data_col = data[data['City'] == col]
            g = sbn.lineplot(data=data_col, x='To Date', y='AQI', ax=ax[q])
            g.set_xlabel(col)
        plt.savefig(aqi_impact_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path = parsed_args.config, params_path = parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
        
        