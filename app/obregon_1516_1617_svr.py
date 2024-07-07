from machines.predictions.svr_prediction import SVR_Prediction
from machines.enums import SvrKernelEnum
from preprocesses.preprocess import Preprocess
from preprocesses.enums import TransformEnum, TypeFileEnum, StandardScaleEnum
from metrics.error_metric import ErrorMetric
from metrics.enums import MetricEnum, PlotLegends
import pandas as pd

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

kernels = [
    SvrKernelEnum.LINEAR,
    SvrKernelEnum.POLY,
    SvrKernelEnum.RBF,
    SvrKernelEnum.SIGMOID,
]


def do_run(do_basic: bool, do_heatmap: bool, do_pca: bool, is_debug:bool = False):
    data = []
    if is_debug:
        print("Start - work")
    for kernel in kernels:
        if do_basic:
            for file_csv in files_csv_original:
                data = run_svr(reductionName="Without",
                               file_csv=file_csv, kernel=kernel, data=data)
        if do_heatmap:
            for file_csv in files_csv_heatmap:
                data = run_svr(reductionName="heatmap",
                               file_csv=file_csv, kernel=kernel, data=data)
        if do_pca:
            for file_csv in files_csv_original:
                data = run_svr(reductionName="pca", file_csv=file_csv, kernel=kernel,
                               data=data, do_pca=True)
    df = pd.DataFrame(
        data, columns=['ML', 'Param','Reduction Technic' ,'Dataset', 'RMSE', 'R2', 'MAPE'])
    df.to_csv("result_svr.csv", index=False)


def run_svr(reductionName: str, file_csv: str, kernel, data, do_pca: bool = False):
    datasetName = f"{
        file_csv
        .replace("data/obregon_1516_1617/", "")
        .replace("o_57_c_", "")
        .replace("o_57_o_", "")
        .replace(".csv", "")
        .replace("_", " ")}"
    ml = SVR_Prediction(kernel=kernel)
    preprocessing = Preprocess(
        file_name=file_csv, type_file=TypeFileEnum.CSV)
    if do_pca:
        preprocessing.read_file(
            transform=TransformEnum.PCA,
            standard_scale=StandardScaleEnum.BASIC,
        )
    else:
        preprocessing.read_file(
            transform=TransformEnum.MEAN,
            standard_scale=StandardScaleEnum.BASIC,
        )
    preprocessing.build_train_and_test()
    x_train, y_train = preprocessing.get_train()
    x_test, y_test = preprocessing.get_test()

    ml.training(x_train=x_train, y_train=y_train)
    y_predicted = ml.prediction(x_test=x_test)
    metric = ErrorMetric(y_predicted=y_predicted,
                         y_test=y_test, x_test=x_test)
    metric.calculate_metric_prediction()
    data.append([
        "SVR",
        kernel.value,
        reductionName,
        datasetName,
        metric.get_metric(metric=MetricEnum.ROOT_MEAN_SQUARED_ERROR),
        metric.get_metric(metric=MetricEnum.R2_SCORE),
        metric.get_metric(metric=MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR),
    ])
    return data


if __name__ == "__main__":
    do_run(do_basic=True, do_heatmap=True, do_pca=True)
