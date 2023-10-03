from svm import calculate_svm_simple, calculate_svm_principal_component_analysis


def do_run():
    result = calculate_svm_simple(file_name='data/6x299_yield_to_machine.csv')
    print("SVM with all variables")
    result.print_results()
    result = calculate_svm_principal_component_analysis(
        file_name='data/6x299_yield_to_machine.csv',
        debug=False,
        n_components=5
    )
    print("SVM with PCA")
    result.print_results()


if __name__ == "__main__":
    do_run()
