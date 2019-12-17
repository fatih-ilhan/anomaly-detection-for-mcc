import numpy as np

import sklearn.metrics as skmetrics


def evaluate(pred, label, num_classes, metric_list):
    result_dict = {}

    label_one_hot = np.zeros((label.shape[0], num_classes))
    label_one_hot[np.arange(label_one_hot.shape[0]), label] = 1

    for metric in metric_list:
        if metric == "confusion_matrix":
            result = skmetrics.confusion_matrix(label, pred.argmax(axis=1))
            result_dict[metric] = np.array(result)
        elif metric == "accuracy":
            result = np.around(skmetrics.accuracy_score(label, pred.argmax(axis=1)) * 100, 2)
            result_dict[metric] = result
        elif metric == "balanced_accuracy":
            result = np.around(skmetrics.balanced_accuracy_score(label, pred.argmax(axis=1)) * 100, 2)
            result_dict[metric] = result
        elif metric == "f1_macro":
            result = np.around(skmetrics.f1_score(label, pred.argmax(axis=1), average="macro") * 100, 2)
            result_dict[metric] = result
        elif metric == "f1_micro":
            result = np.around(skmetrics.f1_score(label, pred.argmax(axis=1), average="micro") * 100, 2)
            result_dict[metric] = result
        elif metric == "roc_auc_macro":
            result = np.around(skmetrics.roc_auc_score(label_one_hot, pred, average="macro") * 100, 2)
            result_dict[metric] = result
        elif metric == "roc_auc_micro":
            result = np.around(skmetrics.roc_auc_score(label_one_hot, pred, average="micro") * 100, 2)
            result_dict[metric] = result
        else:
            raise KeyError

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
