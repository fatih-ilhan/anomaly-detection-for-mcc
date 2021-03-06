import itertools

NUM_VAL_SPLITS = 3
NUM_TEST_SPLITS = 3
N_JOBS = 4
VAL_METRIC = "f1_macro"
EVALUATION_METRIC_LIST = ["f1_macro", "accuracy", "balanced_accuracy", "roc_auc_macro"]

MODEL_PARAMS = {"mlp": {"hidden_layer_sizes": [(100,), (20, 5,), ],
                        "activation": ["relu", "logistic", ],
                        "alpha": [1e-4, 1e-2, ],
                        "max_iter": [250, 1000, ]},
                "random_forest": {"max_depth": [None, 5],
                                  "n_estimators": [10, ]},
                "lsvc": {"C": [10, 1, 1e-1, ],
                              "max_iter": [100, 1000, ]},
                "lof": {"n_neighbors": [5, 20, ],
                        "n_jobs": [N_JOBS, ]},
                "ocsvm": {"kernel": ["linear", "rbf", ],
                          "max_iter": [-1, 100, ]},
                "isolation": {"n_estimators": [20, ],
                              "max_features": [0.5, 1, ],
                              "bootstrap": [False, True, ],
                              "n_jobs": [N_JOBS, ]}
               }
