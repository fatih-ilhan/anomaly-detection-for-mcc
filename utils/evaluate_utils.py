import numpy as np

import sklearn.metrics as skmetrics


def evaluate(pred, label, num_classes, metric_list):
    result_dict = {}

    label_one_hot = np.zeros((label.shape[0], num_classes))
    label_one_hot[np.arange(label_one_hot.shape[0]), label] = 1

    if pred.ndim == 1:
        pred_discrete = (pred >= 0).astype(int)
        label_one_hot = label_one_hot[:, 1]
    else:
        pred_discrete = pred.argmax(axis=1)

    for metric in metric_list:
        if metric == "confusion_matrix":
            result = skmetrics.confusion_matrix(label, pred_discrete)
            result_dict[metric] = np.array(result)
        elif metric == "accuracy":
            result = skmetrics.accuracy_score(label, pred_discrete) * 100
            result_dict[metric] = result
        elif metric == "balanced_accuracy":
            result = skmetrics.balanced_accuracy_score(label, pred_discrete) * 100
            result_dict[metric] = result
        elif metric == "f1_macro":
            result = skmetrics.f1_score(label, pred_discrete, average="macro") * 100
            result_dict[metric] = result
        elif metric == "f1_micro":
            result = skmetrics.f1_score(label, pred_discrete, average="micro") * 100
            result_dict[metric] = result
        elif metric == "roc_auc_macro":
            result = skmetrics.roc_auc_score(label_one_hot, pred, average="macro") * 100
            result_dict[metric] = result
        elif metric == "roc_auc_micro":
            result = skmetrics.roc_auc_score(label_one_hot, pred, average="micro") * 100
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

        out_dict[key] = np.around(out_value, 2)

    return out_dict
