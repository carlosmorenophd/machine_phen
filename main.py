from svm import calculate_svm


def do_run():
    result = calculate_svm(file_name='data/6x299_yield_to_machine.csv')
    print(result.confusion_matrix)
    print(result.accuracy_score)


if __name__ == "__main__":
    do_run()
