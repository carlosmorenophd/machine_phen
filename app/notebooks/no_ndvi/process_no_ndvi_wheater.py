#!/usr/bin/env python
# coding: utf-8

# In[61]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[62]:


# import os
# import sys

# module_path = os.path.abspath(os.path.join("../.."))
# if module_path not in sys.path:
#     sys.path.append(module_path)


# In[63]:


from machines.predictions import RF_Prediction, SVR_Prediction, XGB_Prediction
from machines.enums import SvrKernelEnum
from preprocesses.preprocess import Preprocess
from preprocesses.enums import TransformEnum, TypeFileEnum, StandardScaleEnum
from metrics.error_metric import ErrorMetric
from metrics.enums import MetricEnum
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json


# In[64]:


# In[65]:


def get_machine_metric(machine, file_csv):
    preprocessing = Preprocess(file_name=file_csv, type_file=TypeFileEnum.CSV)
    preprocessing.read_file(
        transform=TransformEnum.MEAN,
        standard_scale=StandardScaleEnum.BASIC,
    )
    preprocessing.build_train_and_test()
    x_train, y_train = preprocessing.get_train()

    # machine = RF_Prediction(n_estimators=1000)
    machine.training(x_train=x_train, y_train=y_train)

    x_test, y_test = preprocessing.get_test()
    y_predicted = machine.prediction(x_test=x_test)

    metric = ErrorMetric(y_predicted=y_predicted, y_test=y_test, x_test=x_test)
    metric.calculate_metric_prediction()
    return (
        metric.get_metric(metric=MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR),
        metric.get_metric(metric=MetricEnum.ROOT_MEAN_SQUARED_ERROR),
        metric.get_metric(metric=MetricEnum.R2_SCORE),
    )


# In[66]:


class CountProgress:
    def __init__(self, file_base: str = "progress_store") -> None:
        self.progress_json = f"test_{file_base}_progress.csv"
        self.end_json = f"test_{file_base}_end.csv"

    def set_total(self, total: int) -> None:
        with open(self.end_json, "w") as f:
            json.dump(total, f)
        with open(self.progress_json, "w") as f:
            json.dump(0, f)

    def increase(self):
        try:
            with open(self.progress_json, "r") as f:
                count = json.load(f)
                count += 1
            with open(self.progress_json, "w") as f:
                json.dump(count, f)
        except FileNotFoundError:
            return 0


# In[67]:


import itertools


def do_run():
    file_csv = "data/obregon_1516_1617/original/o_57_o_phenotypic_weather_d.csv"
    # print(os.getcwd())
    df = pd.read_csv(file_csv)

    columns_no_ndvi = []
    for col in df.columns:
        if "ndvi" not in col.lower():
            columns_no_ndvi.append(col)

    print(len(columns_no_ndvi))
    new_csv_file = "test_no_ndvi.csv"
    df.to_csv(new_csv_file, index=False)
    # print(columns_no_ndvi)
    columns_to_iterate = columns_no_ndvi[1:-1]
    column_y = columns_no_ndvi[-1]

    # print(columns_to_iterate, column_y)

    columns_selection = [
        x for x in itertools.product([True, False], repeat=len(columns_to_iterate))
    ][:-1]
    progress = CountProgress()
    mape = 1
    best = []
    metrics = []
    machines = ["RF", "SVR_LINEAL", "XGB"]
    progress.set_total(total=len(columns_selection) * len(machines))
    for columns in columns_selection:
        selected = np.array(columns_to_iterate)[np.array(columns)].tolist()
        selected.append(column_y)
        file_csv = "test_selection_characteristic.csv"
        df[selected].to_csv(file_csv, index=False)
        for machine_alias in machines:
            if machine_alias == "RF":
                machine = RF_Prediction(n_estimators=1000)
            elif machine_alias == "SVR_LINEAL":
                machine = SVR_Prediction(kernel=SvrKernelEnum.LINEAR)
            elif machine_alias == "XGB":
                machine = XGB_Prediction()

            print(f"start: {machine_alias} - {len(selected)}")

            new_mape, r_s_m_e, r2 = get_machine_metric(
                machine=machine, file_csv=file_csv
            )
            metrics.append([",".join(selected), machine_alias, new_mape, r_s_m_e, r2])
            progress.increase()
            if new_mape < mape:
                mape = new_mape
                best = selected

    print(mape)
    data = pd.DataFrame(data=metrics, columns=["variable", "ml", "mape", "rsme", "r2"])
    data.to_csv("test_result_all.csv", index=False)
