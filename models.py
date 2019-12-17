import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import OneClassSVM
from pyod.models.lof import LOF
from sklearn.ensemble import IsolationForest


ad_dispatcher = {"ocsvm": OneClassSVM,
                 "lof": LOF,
                 "isolation": IsolationForest}

score_func_dispatcher = {"ocsvm": "decision_function",
                         "lof": "predict_proba",
                         "isolation": "decision_function"}


class AnomalyDetector(BaseEstimator, TransformerMixin):

    def __init__(self, model_name="lof", mode="adcwf", num_classes=1, params_dict=None):
        self.model_name = model_name
        self.mode = mode
        self.num_classes = num_classes

        if params_dict is None:
            params_dict = {}

        params = params_dict.get(model_name, {})

        self.detectors = []
        for _ in range(num_classes):
            self.detectors.append(ad_dispatcher[model_name](**params))

        self.fitted_ = False

    def fit(self, x, y):

        for i in range(self.num_classes):
            mask = y == i
            self.detectors[i].fit(x[mask, :])

        self.fitted_ = True

        return self

    def transform(self, x):

        anomaly_scores = np.zeros((x.shape[0], self.num_classes), dtype=np.float)

        for i, anomaly_detector in enumerate(self.detectors):
            try:
                anomaly_scores[:, i] = getattr(anomaly_detector, score_func_dispatcher[self.model_name])(x)[:, 1]
            except:
                anomaly_scores[:, i] = getattr(anomaly_detector, score_func_dispatcher[self.model_name])(x)

        if self.mode == "adc":
            out = np.concatenate([x, anomaly_scores], axis=1)
        else:
            out = anomaly_scores

        return out
