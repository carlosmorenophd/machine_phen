#!/usr/bin/env python
# coding: utf-8


# In[2]:


import os
import sys
import itertools

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[3]:


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


# In[5]:


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


# In[6]:


class CountProgress:
    def __init__(self, file_base: str = "progress_store") -> None:
        self.progress_json = f"test_{file_base}_progress.csv"
        self.end_json = f"test_{file_base}_end.csv"
        
    def set_total(self, total: int) -> None:
        with open(self.end_json, 'w') as f:
            json.dump(total, f)
        with open(self.progress_json, 'w') as f:
            json.dump(0, f)

    def increase(self):
        try:
            with open(self.progress_json, 'r') as f:
                count = json.load(f) 
                count += 1
            with open(self.progress_json, 'w') as f:
                json.dump(count, f)    
        except FileNotFoundError:
            return 0


# In[10]:

def run ():
    print(os.getcwdb())

    files = [
        # {
        #     "file": "../../../data/obregon_1516_1617/original/o_57_o_phenotypic_weather_m.csv",
        #     "key": "Monthly",
        # },
        # {
        #     "file": "../../../data/obregon_1516_1617/original/o_57_o_phenotypic_weather_w.csv",
        #     "key": "Weekly",
        # },
        # {
        #     "file": "../../../data/obregon_1516_1617/original/o_57_o_phenotypic_weather_f.csv",
        #     "key": "Fifthly",
        # },
        # {
        #     "file": "../../../data/obregon_1516_1617/original/o_57_o_phenotypic_weather_d.csv",
        #     "key": "Daily",
        # },
        {
            "file": "../data/obregon_1516_1617/original/o_57_o_phenotypic.csv",
            "key": "None",
        },
    ]
    mape = 1
    best = []
    metrics = []
    machines = ["RF", "SVR_LINEAL", "XGB"]
    for file in files:
        print(file)
        df = pd.read_csv(file["file"])

        columns_ndvi = []
        for col in df.columns:
            if "ndvi" in col.lower():
                columns_ndvi.append(col)
        columns_best = ['HI', 'BM', 'CT_UAV_1']
        columns_to_iterate = columns_ndvi
        column_y = "YLD"
        columns_selection = [
            x for x in itertools.product([True, False], repeat=len(columns_to_iterate))
        ][:-1]
        progress = CountProgress()
        progress.set_total(total=len(columns_selection) * len(machines))
        for columns in columns_selection:
            selected = np.array(columns_to_iterate)[np.array(columns)].tolist()
            selected = selected + columns_best 
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

                new_mape, r_s_m_e, r2 = get_machine_metric(
                    machine=machine, file_csv=file_csv
                )
                metrics.append(
                    [
                        "-",
                        "-".join(np.array(columns_to_iterate)
                                [np.array(columns)].tolist()),
                        file["key"],
                        machine_alias,
                        new_mape,
                        r_s_m_e,
                        r2,
                    ]
                )
                progress.increase()
                if new_mape < mape:
                    mape = new_mape
                    best = selected

    print(mape, best)
    data = pd.DataFrame(
        data=metrics,
        columns=[
            "all variable",
            "variable cultivo",
            "variable climatica",
            "ml",
            "mape",
            "R_S_M_E",
            "r2",
        ],
    )
    data.to_csv("test_result_all.csv", index=False)

