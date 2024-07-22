# from machines.predictions import RF_Prediction
# from preprocesses.preprocess import Preprocess
# from preprocesses.enums import TransformEnum, TypeFileEnum, StandardScaleEnum
# from metrics.error_metric import ErrorMetric
# from metrics.enums import MetricEnum
# from typing import List
# import pandas as pd
# from os import path
from do_run.run_machine import Do_Run_ML
from machines.predictions import SVR_Prediction, SvrKernelEnum

files_csv_original = [
    "data/obregon_1516_1617/original/o_57_o_phenotypic.csv",
    "data/obregon_1516_1617/original/o_57_o_phenotypic_weather_d.csv",
    "data/obregon_1516_1617/original/o_57_o_phenotypic_weather_w.csv",
    "data/obregon_1516_1617/original/o_57_o_phenotypic_weather_f.csv",
    "data/obregon_1516_1617/original/o_57_o_phenotypic_weather_m.csv",
]

files_csv_heatmap = [
    "data/obregon_1516_1617/correlation/o_57_c_phenotypic.csv",
    "data/obregon_1516_1617/correlation/o_57_c_phenotypic_weather_d.csv",
    "data/obregon_1516_1617/correlation/o_57_c_phenotypic_weather_w.csv",
    "data/obregon_1516_1617/correlation/o_57_c_phenotypic_weather_f.csv",
    "data/obregon_1516_1617/correlation/o_57_c_phenotypic_weather_m.csv",
]


def launch_all():
    do = Do_Run_ML(file_name_save="run_all_ml_test.csv")
    do.adding_dataset(
        key="original",
        names=files_csv_original,
        is_search_best_pca_component=False,
    )
    do.adding_dataset(
        key="heatmap",
        names=files_csv_original,
        is_search_best_pca_component=False,
    )
    do.adding_dataset(
        key="pca",
        names=files_csv_original,
        is_search_best_pca_component=True,
    )
    do.adding_machine_dataset(
        machine=SVR_Prediction(kernel=SvrKernelEnum.LINEAR), key_dataset="original"
    )
    do.run()
