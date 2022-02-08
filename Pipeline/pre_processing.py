import numpy as np
import pandas as pd
import mlflow
import warnings
import os


def one_hot_encode(data, col, top_x):
    labels = [x for x in data[col].value_counts().sort_values(ascending=False).head(top_x).index]
    for label in labels:
        data[col + '_' + label] = np.where(data[col] == label, 1, 0)


def part_date_time(time_s: str):
    date, time = time_s.split(' ')
    day = int(date.split('-')[-1])
    hour = int(time.split(':')[0])
    return pd.Series({'click_day': day, 'click_hour': hour})


def feat_eng():
    df = pd.read_csv("Data/train_dataset.csv")

    """
        Dropping useless columns and those that would give us 100% accuracy like SalesAmountInEuro.
    """
    df_cleaned = df.drop(['SalesAmountInEuro', 'time_delay_for_conversion', 'product_price', 'audience_id', 'user_id',
                          'product_category(7)'], axis=1)
    df_cleaned = df_cleaned.replace(['-1', -1], np.nan)

    """
        Our Dataset has lots of null values. To overcome this issue to some extent,
        We will remove rows with null values equal to or more than a threshold.
    """
    threshold = 11
    to_be_removed_indx = []
    for index, row in df_cleaned.iterrows():
        nulls = 0
        for i in row.keys():
            if pd.isnull(row[i]):
                nulls += 1
        if nulls >= threshold:
            to_be_removed_indx.append(index)

    df_clean = df_cleaned.drop(to_be_removed_indx)

    """
        We have a 'timestamp' column. We'll separate that into two different columns,
        So we can work with it.
    """


    df_clean = df_clean.merge(df_clean['click_timestamp'].apply(lambda t: part_date_time(t)), left_index=True,
                              right_index=True)
    df_clean = df_clean.drop('click_timestamp', axis=1)

    """
        Categorical columns should be one-hot.
    """

    numerical_columns = df_clean.select_dtypes(exclude='object')
    categorical_columns = df_clean.select_dtypes(include='object')



    columns = list(categorical_columns.columns)
    for col in columns:
        one_hot_encode(df_clean, col, min(250, df_clean[col].nunique()))
    for col in columns:
        df_clean = df_clean.drop(col, axis=1)


    """
        We no longer have categotical columns. the column 'nb_clicks_1week'
        Has some null values which we'll replace with mean.
    """

    df_clean['nb_clicks_1week'].fillna((df_clean['nb_clicks_1week'].mean()), inplace=True)

    """
        Saving cleaned DataFrame
    """
    df_clean.to_csv("Data/clean_dataset.csv", index=False)


class mlModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        feat_eng()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    with mlflow.start_run() as runner:
        model = mlModel()
        model_path = os.path.join('mlflow_models', "pre_processing_"+runner.info.run_id)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)
        reload_model = mlflow.pyfunc.load_model(model_path)
        print(f'runner is: {runner.info.run_id}')
