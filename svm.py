# from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model import ResultSVM


def calculate_svm(file_name: str, debug: bool = False) -> ResultSVM:
    """_summary_ Calculate SVM return matrix confusion and accuracy
    """
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    if debug:
        print("Variable X =>", X)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X)
    X = imputer.transform(X)

    if debug:
        print("Transform variable X =>", X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    if debug:
        print("Training data => ", X_train)
        print("Training y => ", y_train)
        print("Set to test => ", X_test)
        print("Test y => ", y_test)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if debug:
        print("Scaling training set =>", X_train)
        print("Scaling test set =>", X_test)

    # Training the Kernel SVM model on the Training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting a new result
    # print(classifier.predict(sc.transform([[38.6, 61,	4,	109,	149,	83,
    #                                         ]])))

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    if debug:
        print("Result of testing => ")
        print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                              y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    if debug:
        print("Confusion matrix => ", cm)
        print("Accuracy => ", ac)
    result = ResultSVM(confusion_matrix=cm, accuracy_score=ac)
    return result

# # Visualizing the Training set results
# X_set, y_set = sc.inverse_transform(X_train), y_train
# X1, X2, X3, X4, X5, X6 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10,
#                                stop=X_set[:, 0].max() + 10, step=0.25),
#                      np.arange(start=X_set[:, 1].min() - 1000,
#                                stop=X_set[:, 1].max() + 1000, step=0.25),
#                      np.arange(start=X_set[:, 2].min() - 1000,
#                                stop=X_set[:, 2].max() + 1000, step=0.25),
#                      np.arange(start=X_set[:, 3].min() - 1000,
#                                stop=X_set[:, 3].max() + 1000, step=0.25),
#                      np.arange(start=X_set[:, 4].min() - 1000,
#                                stop=X_set[:, 4].max() + 1000, step=0.25),
#                      np.arange(start=X_set[:, 5].min() - 1000,
#                                stop=X_set[:, 5].max() + 1000, step=0.25),
#                      )
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('Kernel SVM (Training set)')
# plt.xlabel('Trait')
# plt.ylabel('Yield')
# plt.legend()
# plt.show()

# # Visualizing the Test set results
# X_set, y_set = sc.inverse_transform(X_test), y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10,
#                                stop=X_set[:, 0].max() + 10, step=0.25),
#                      np.arange(start=X_set[:, 1].min() - 1000,
#                                stop=X_set[:, 1].max() + 1000, step=0.25)
#                      )
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('Kernel SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
