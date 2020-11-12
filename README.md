# anomaly-detection-for-mcc
This repository contains the code for:

F. Ilhan, S. F. Yilmaz and S. S. Kozat, “A Two-Stage Multi-Class Classification Approach Based on Anomaly Detection”, 28th IEEE Signal Processing and Communications Applications, 2020.

## Run the code:
To run the experiments, give the following arguments to main.py:

```python
parser.add_argument('--mode_list', type=str, nargs='+')  # choose items from "normal", "adcwf", "adc
parser.add_argument('--ad_list', type=str, nargs='+')  # choose items from "ocsvm", "lof", "isolation"
parser.add_argument('--mcc_list', type=str, nargs='+')  # choose items from "mlp", "random_forest", "lsvc"
parser.add_argument('--dataset_list', type=str, nargs='+')  # choose items from "smartphone", "breast_cancer", "digits", "iris", "wine"
parser.add_argument('--num_repeat', type=int, default=1)  # repeat train + test
```
You can set model, anomaly detector, data and simulation parameter loops from `config.py`.
