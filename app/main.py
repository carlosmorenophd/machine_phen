from preprocessing.preprocessing import Preprocessing
from preprocessing.enums import TypeFileEnum, TransformEnum
from preprocessing.pca_preprocessing import PCA_Preprocessing


def do_run():
    get_data = Preprocessing(file_name="/home/yeiden/repo/machine_learning_phenotypic/dataset.csv",
                             type_file=TypeFileEnum.CSV, is_debug=True)
    get_data.read_file(transform=TransformEnum.MEAN)
    pca = PCA_Preprocessing(data=get_data.get_x_transform(), is_debug=True)
    most_important_columns, _ = pca.evaluate_pca(n_components=30)
    print(get_data.get_name_of_column_by_index(most_important_columns))
    


if __name__ == "__main__":
    do_run()
