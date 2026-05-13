# model building
import pandas as pd
import os
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier
from src.data.data_collection import load_data, read_params, split_data, save_data


def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        x_train = data.drop(columns = ["Potability"], axis=1)
        y_train = data["Potability"]
        return x_train,y_train
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")
def train_model(x : pd.DataFrame, y : pd.date_range, n_estimators : int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators)
        clf.fit(x,y)
        return clf
    except Exception as e:
        raise Exception(f"Error in taining model data:{e}")
def store_model(model : RandomForestClassifier,filepath: str):
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error in storing model data:{e}")
def main():
    data_pathfile = "./data/processed/"
    params_file = "params.yaml"
    model_name = "models/model.pkl"
    try:
        train_data = load_data(os.path.join(data_pathfile,"train_processed.csv"))
        x_train,y_train = prepare_data(train_data)
        params = read_params(params_file)
        n_estimators = params["model_building"]["n_estimators"]
        model = train_model(x_train,y_train, n_estimators)
        store_model(model,model_name)
    except Exception as e:
        raise Exception (f"Error in main file occured:{e}")
if __name__ == "__main__":
    main()