from preprocessing.preprocessing import Preprocessing
from preprocessing.enums import TypeFileEnum, TransformEnum
from preprocessing.pca_preprocessing import PCA_Preprocessing


def do_run():
    get_data = Preprocessing(file_name="/home/yeiden/repo/machine_learning_phenotypic/dataset.csv",
                             type_file=TypeFileEnum.CSV, is_debug=True)
    get_data.read_file(transform=TransformEnum.MEAN)
    pca_preprocessing = PCA_Preprocessing(
        data=get_data.get_x_transform(), target= get_data.get_y(), feature_names= get_data.get_name_features(), is_debug=True)
    pca_preprocessing.evaluate_pca()
    # pca_preprocessing.graph_sedimentation()
    # pca_preprocessing.graph_scores()
    # pca_preprocessing.graph_influence()
    # pca_preprocessing.graph_projection()
    pca_preprocessing.graph_outlier()

if __name__ == "__main__":
    do_run()
