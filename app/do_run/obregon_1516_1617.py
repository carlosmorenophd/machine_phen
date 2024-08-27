from do_run.run_machine import Do_Run_ML
from machines.predictions import (
    SVR_Prediction,
    SvrKernelEnum,
    XGB_Prediction,
    RF_Prediction,
)

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

files_csv_no_ndvi = [
    'data/obregon_1516_1617/original/o_57_o_phenotypic_no_ndvi.csv'
]


def adding_svr(do):
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.LINEAR), key_dataset="original"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.POLY), key_dataset="original"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.RBF), key_dataset="original"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.SIGMOID), key_dataset="original"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.LINEAR), key_dataset="heatmap"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.POLY), key_dataset="heatmap"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.RBF), key_dataset="heatmap"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.SIGMOID), key_dataset="heatmap"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.LINEAR), key_dataset="pca"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.POLY), key_dataset="pca"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.RBF), key_dataset="pca"
    # )
    # do.adding_machine_dataset(
    #     machine=SVR_Prediction(kernel=SvrKernelEnum.SIGMOID), key_dataset="pca"
    # )

    do.adding_machine_dataset(
        machine=SVR_Prediction(kernel=SvrKernelEnum.LINEAR), key_dataset="no_ndvi"
    )
    do.adding_machine_dataset(
        machine=SVR_Prediction(kernel=SvrKernelEnum.POLY), key_dataset="no_ndvi"
    )
    do.adding_machine_dataset(
        machine=SVR_Prediction(kernel=SvrKernelEnum.RBF), key_dataset="no_ndvi"
    )
    do.adding_machine_dataset(
        machine=SVR_Prediction(kernel=SvrKernelEnum.SIGMOID), key_dataset="no_ndvi"
    )

    return do


def adding_xgb(do):
    # do.adding_machine_dataset(machine=XGB_Prediction(), key_dataset="original")
    # do.adding_machine_dataset(machine=XGB_Prediction(), key_dataset="heatmap")
    # do.adding_machine_dataset(machine=XGB_Prediction(), key_dataset="pca")
    do.adding_machine_dataset(machine=XGB_Prediction(), key_dataset="no_ndvi")

    return do


def adding_rf(do):
    # do.adding_machine_dataset(machine=RF_Prediction(), key_dataset="original")
    # do.adding_machine_dataset(machine=RF_Prediction(), key_dataset="heatmap")
    # do.adding_machine_dataset(machine=RF_Prediction(), key_dataset="pca")
    do.adding_machine_dataset(machine=RF_Prediction(), key_dataset="no_ndvi")
    return do


def launch_all(is_debug: bool = False):
    do = Do_Run_ML(file_name_save="run_all_ml_test.csv", is_debug=is_debug)
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
    do.adding_dataset(
        key="no_ndvi",
        names=files_csv_no_ndvi,
        is_search_best_pca_component=True,
    )
    do = adding_svr(do=do)
    do = adding_rf(do=do)
    do = adding_xgb(do=do)
    do.run()
