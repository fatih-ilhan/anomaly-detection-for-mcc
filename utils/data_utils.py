import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine


"""
Includes data utilization functions
"""


def prepare_data(dataset_name):
    """
    :param dataset_name: str
    :return: dict of x_train (n_train x d), y_train (n_train), x_test (n_test x d), y_test (n_test)
             If no test split is predefined, returns None for them
    """

    helper_dispatcher = {"smartphone": prepare_data_smartphone,
                         "digits": prepare_data_sklearn,
                         "breast_cancer": prepare_data_sklearn,
                         "iris": prepare_data_sklearn,
                         "wine": prepare_data_sklearn,
                         }

    args = []
    if dataset_name in ["digits", "breast_cancer", "iris", "wine"]:
        args.append(dataset_name)

    data = helper_dispatcher[dataset_name](*args)

    data_dict = {"x_train": data[0],
                 "y_train": data[1],
                 "x_test": None,
                 "y_test": None}

    if len(data) > 2:
        data_dict["x_test"] = data[2]
        data_dict["y_test"] = data[3]

    return data_dict


def prepare_data_sklearn(dataset_name):
    if dataset_name == "iris":
        return load_iris(return_X_y=True)
    if dataset_name == "digits":
        return load_digits(return_X_y=True)
    if dataset_name == "breast_cancer":
        return load_breast_cancer(return_X_y=True)
    if dataset_name == "wine":
        return load_wine(return_X_y=True)


def prepare_data_smartphone():
    x_train_data_path = os.path.join("data", "smartphone_activity", "Train", "X_train.txt")
    x_train_df = pd.read_csv(x_train_data_path, header=None, sep=" ")
    x_train = np.array(x_train_df)

    y_train_data_path = os.path.join("data", "smartphone_activity", "Train", "y_train.txt")
    y_train_df = pd.read_csv(y_train_data_path, header=None, sep=" ")
    y_train = np.array(y_train_df)[:, 0] - 1

    x_test_data_path = os.path.join("data", "smartphone_activity", "Test", "X_test.txt")
    x_test_df = pd.read_csv(x_test_data_path, header=None, sep=" ")
    x_test = np.array(x_test_df)

    y_test_data_path = os.path.join("data", "smartphone_activity", "Test", "y_test.txt")
    y_test_df = pd.read_csv(y_test_data_path, header=None, sep=" ")
    y_test = np.array(y_test_df)[:, 0] - 1

    return x_train, y_train, x_test, y_test
