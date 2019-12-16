NUM_VAL_SPLITS = 3
NUM_TEST_SPLITS = 3
N_JOBS = 4
VAL_METRIC = "balanced_accuracy"
EVALUATION_METRIC_LIST = ["balanced_accuracy", "f1"]


MODEL_PARAMS = {"mlp": {"hidden_layer_sizes": [(100,), (20, 5,), ],
                                  "activation": ["relu", "logistic", ],
                                  "alpha": [1e-4, 1e-2, ],
                                  "max_iter": [250, 1000, ],
                                  },
                "random_forest": {"max_depth": [None, 5],
                                           "n_estimators": [10, 30, 100, ]},
                "lsvc": {"C": [10, 1, 1e-1, ],
                              "max_iter": [10, 100, 1000, ]},
                "lof": {},
                "ocsvm": {},
                "isolation": {}
               }

