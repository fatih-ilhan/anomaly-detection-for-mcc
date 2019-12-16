NUM_VAL_SPLITS = 3
NUM_TEST_SPLITS = 3
NUM_REPEAT = 5
N_JOBS = 4


MODEL_PARAMS = {"MLPClassifier": {"hidden_layer_sizes": [(100,), (20, 5,), ],
                                  "activation": ["relu", "logistic", ],
                                  "alpha": [1e-4, 1e-2, ],
                                  "max_iter": [250, 1000, ],
                                  },
                "RandomForestClassifier": {"max_depth": [None, 5],
                                           "n_estimators": [10, 30, 100, ]},
                "LinearSVC": {"C": [10, 1, 1e-1, ],
                              "max_iter": [10, 100, 1000, ]},
                "LOF": {},
                "OneClassSVM": {},
                "IsolationForest": {}
                }