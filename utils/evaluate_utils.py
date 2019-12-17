import numpy as np

import sklearn.metrics as skmetrics


def evaluate(pred, label, metric_list):
    result_dict = {}

    for metric in metric_list:
        if metric == "confusion_matrix":
            result = skmetrics.confusion_matrix(label, pred)
            result_dict[metric] = np.array(result)
        elif metric == "accuracy":
            result = np.around(skmetrics.accuracy_score(label, pred) * 100, 2)
            result_dict[metric] = result
        elif metric == "balanced_accuracy":
            result = np.around(skmetrics.balanced_accuracy_score(label, pred) * 100, 2)
            result_dict[metric] = result
        elif metric == "balanced_f1":
            result = np.around(skmetrics.f1_score(label, pred, average="weighted") * 100, 2)
            result_dict[metric] = result

    return result_dict


def merge_results(results_dict_list, mode="mean"):
    out_dict = {}

    if len(results_dict_list) < 1:
        return out_dict

    if len(set([len(results_dict.keys()) for results_dict in results_dict_list])) > 1:
        raise ValueError

    for key in results_dict_list[0].keys():
        values = [results_dict[key] for results_dict in results_dict_list]
        if mode == "mean":
            out_value = sum(values) / len(values)
        elif mode == "std":
            out_value = np.std(values, axis=0)
        else:
            raise NotImplementedError

        out_dict[key] = out_value.tolist()

    return out_dict
