# data processing
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
from src.data.data_collection import load_data, read_params, split_data, save_data

def missing_vlaues_meadian(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        return df
    except Exception as e:
        raise Exception ("Error in filling missing values:{e}")
def main():
    raw_data_path = "./data/raw/"
    processed_data_path = "./data/processed" 
    try:
        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))
        train_processed_data = missing_vlaues_meadian(train_data)
        test_processed_data = missing_vlaues_meadian(test_data) 
        os.makedirs(processed_data_path, exist_ok=True)
        save_data(train_processed_data, os.path.join(processed_data_path,"train_processed.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path,"test_processed.csv"))
    except Exception as e:
        print(f"Error in data processing process: {e}")


if __name__ == "__main__":
    main()