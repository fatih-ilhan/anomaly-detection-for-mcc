import os
import sys
import warnings
import json
import argparse

import numpy as np
from pyod.models.lof import LOF
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
from utils.data_utils import prepare_data
from utils.evaluate_utils import evaluate, merge_results
from models import AnomalyDetector

# very harsh way to ignore warnings, but i don't care
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

random_seed = 5

ad_dispatcher = {"ocsvm": OneClassSVM,
                 "lof": LOF,
                 "isolation": IsolationForest}

mcc_dispatcher = {"mlp": MLPClassifier,
                  "random_forest": RandomForestClassifier,
                  "lsvc": LinearSVC}

score_func_dispatcher = {"mlp": "predict_proba",
                         "ocsvm": "decision_function",
                         "lsvc": "decision_function",
                         "lof": "predict_proba",
                         "random_forest": "predict_proba",
                         "isolation": "decision_function"}


def create_pipeline(ad_name, mcc_name, mode, num_classes):
    ad_params = config.MODEL_PARAMS[ad_name]
    mcc_params = config.MODEL_PARAMS[mcc_name]

    estimator_list = [("std_1", StandardScaler())]
    param_grid = {}

    if mode != "normal":
        detectors = [ad_dispatcher[ad_name]()] * num_classes
        ad = AnomalyDetector(ad_name, mode, num_classes, detectors)
        estimator_list.append((ad_name, ad))
        estimator_list.append(("std_2", StandardScaler()))
        for key, val in ad_params.items():
            param_grid[ad_name+"__"+key] = val

    mcc = mcc_dispatcher[mcc_name]
    estimator_list.append((mcc_name, mcc()))
    for key, val in mcc_params.items():
        param_grid[mcc_name + "__" + key] = val

    model = GridSearchCV(estimator=Pipeline(estimator_list),
                         param_grid=param_grid,
                         cv=config.NUM_VAL_SPLITS,
                         scoring=config.VAL_METRIC,
                         n_jobs=config.N_JOBS)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode_list', type=str, nargs='+')  # "normal", "adcwf", "adc
    parser.add_argument('--ad_list', type=str, nargs='+')
    parser.add_argument('--mcc_list', type=str, nargs='+')
    parser.add_argument('--dataset_list', type=str, nargs='+')  # string list
    parser.add_argument('--num_repeat', type=int, default=1)  # repeat train + test

    args = parser.parse_args()

    model_dispatcher = {}
    params_dispatcher = {}

    ad_list = args.ad_list
    mcc_list = args.mcc_list

    for dataset_name in args.dataset_list:

        print(f"**********{dataset_name}**********")

        data_dict = prepare_data(dataset_name)
        x, y, x_test, y_test = data_dict.values()
        num_classes = len(np.unique(y))

        num_test_splits = config.NUM_TEST_SPLITS
        if x_test is not None:  # Test split is predefined in the dataset
            num_test_splits = 1

        skf = StratifiedKFold(n_splits=num_test_splits, random_state=random_seed)

        for mode in args.mode_list:

            if mode == "normal":
                cur_ad_list = [ad_list[0]]  # don't repeat unnecessarily because we won't use anomaly detectors
            else:
                cur_ad_list = ad_list

            for ad_name in cur_ad_list:

                for mcc_name in args.mcc_list:

                    test_results_list = []

                    for rep in range(args.num_repeat):
                        # print("********************")
                        # print(f"Mode: {mode}, AD: {ad_name}, MCC: {mcc_name}, Dataset: {dataset_name}, Repeat index: {rep + 1}")

                        split_idx = 0

                        while split_idx < num_test_splits:

                            if x_test is None:
                                train_index, test_index = next(skf.split(x, y))
                                cur_x_train = x[train_index]
                                cur_y_train = y[train_index]
                                cur_x_test = x[test_index]
                                cur_y_test = y[test_index]
                            else:
                                cur_x_train = x.copy()
                                cur_y_train = y.copy()
                                cur_x_test = x_test.copy()
                                cur_y_test = y_test.copy()

                            model = create_pipeline(ad_name, mcc_name, mode, num_classes)

                            try:
                                model.fit(cur_x_train, cur_y_train)
                            except Exception as e:
                                print(e)

                            try:
                                test_preds = getattr(model, score_func_dispatcher[mcc_name])(cur_x_test)
                            except Exception as e:
                                print(e)

                            test_results = evaluate(test_preds, cur_y_test, num_classes, config.EVALUATION_METRIC_LIST)
                            test_results_list.append(test_results)

                            split_idx += 1

                    average_test_results_mean = merge_results(test_results_list, "mean")
                    average_test_results_std = merge_results(test_results_list, "std")

                    print(f"Mode: {mode:7}| AD: {ad_name:10}| MCC: {mcc_name:14}| Dataset: {dataset_name:13}| "
                          f"Test results (mean):", average_test_results_mean)
                    print(f"Mode: {mode:7}| AD: {ad_name:10}| MCC: {mcc_name:14}| Dataset: {dataset_name:13}| "
                          f"Test results (std):", average_test_results_std)
