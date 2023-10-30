from svm import (calculate_svm_simple,
                 calculate_svm_principal_component_analysis)
from utils import get_data
from utils import reducer
from sklearn.impute import SimpleImputer

def do_run():
    # result = calculate_svm_simple(file_name='data/6x299_yield_to_machine.csv')
    # print("SVM with all variables")
    # result.print_results()
    # result = calculate_svm_principal_component_analysis(
    #     file_name='data/6x299_yield_to_machine.csv',
    #     debug=False,
    #     n_components=5
    # )
    # print("SVM with PCA")
    # result.print_results()
    X, y = get_data.get_from_csv("/home/yeiden/Downloads/04_trait_dataset_clean.csv")
    print(X.shape, y.shape);
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(X)
    most_important_columns, variance_ratio = reducer.get_by_pca_best_columns(X=data)
    print(most_important_columns)





if __name__ == "__main__":
    do_run()
