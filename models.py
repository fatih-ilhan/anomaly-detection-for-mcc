import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

score_func_dispatcher = {"ocsvm": "decision_function",
                         "lof": "predict_proba",
                         "isolation": "decision_function"}


class AnomalyDetector(BaseEstimator, TransformerMixin):

    def __init__(self, model_name="lof", mode="adcwf", num_classes=1, detectors=[]):
        self.model_name = model_name
        self.mode = mode
        self.num_classes = num_classes
        self.detectors = detectors

    def fit(self, x, y):

        for i in range(self.num_classes):
            mask = y == i
            self.detectors[i].fit(x[mask, :])

        self.fitted_ = True

        return self

    def transform(self, x):

        scores = np.zeros((x.shape[0], self.num_classes), dtype=np.float)

        for i, detector in enumerate(self.detectors):
            try:
                scores[:, i] = getattr(detector, score_func_dispatcher[self.model_name])(x)[:, 1]
            except:
                scores[:, i] = getattr(detector, score_func_dispatcher[self.model_name])(x)

        if self.mode == "adc":
            out = np.concatenate([x, scores], axis=1)
        else:
            out = scores

        return out

    # def get_params(self, deep=True):
    #     return self.__dict__

    def set_params(self, **params):
        for detector in self.detectors:
            detector.set_params(**params)

        return self
