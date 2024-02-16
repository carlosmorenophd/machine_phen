from app.preprocesses.preprocess import Preprocess
from preprocesses.enums import TypeFileEnum, TransformEnum
from app.preprocesses.pca_preprocess import PCA_Preprocess


def do_run():
    get_data = Preprocess(
        file_name="./dataset.csv",
        type_file=TypeFileEnum.CSV, 
        is_debug=True,
    )
    get_data.read_file(transform=TransformEnum.MEAN)
    pca_preprocessing = PCA_Preprocess(
        data=get_data.get_x_transform(), target= get_data.get_y(), feature_names= get_data.get_name_features(), is_debug=True)
    pca_preprocessing.evaluate_pca()
    pca_preprocessing.write_to_csv(csv_name="result_test.csv")
    pca_preprocessing.outlier_excel(xls_name="atipicos_test.xlsx")
    # pca_preprocessing.graph_sedimentation()
    # pca_preprocessing.graph_scores()
    # pca_preprocessing.graph_influence()
    # pca_preprocessing.graph_projection()
    # pca_preprocessing.graph_outlier()

if __name__ == "__main__":
    do_run()
