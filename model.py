from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt


class ResultSVM ():
    """_summary_ Storage all result statistic of SMV
    """

    def __init__(self) -> None:
        self.confusion_matrixes = None
        self.accuracy_score: float = None
        self.sensibility: float = None
        self.specificity: float = None
        self.value_predictive_positive: float = None
        self.value_predictive_negative: float = None
        self.f1_score: float = None
        self.area_under_curve = None
        self.receiver_operating_characteristic = None

    def calculate_all_basic(self, y_test, y_pred):
        self.confusion_matrixes = confusion_matrix(y_test, y_pred)
        self.accuracy_score = accuracy_score(y_test, y_pred)

    def calculate_all_statistic(self, y_test, y_pred):
        self.confusion_matrixes = confusion_matrix(y_test, y_pred)
        self.accuracy_score = accuracy_score(y_test, y_pred)
        self.sensibility = recall_score(y_test, y_pred)
        returned = self.confusion_matrixes.ravel()
        if len(returned) == 4:
            TN, FP, FN, TP = returned
            self.specificity = TN / (TN + FP)
            self.value_predictive_positive = TP / (TP+FP)
            self.value_predictive_negative = TN / (TP + FN)
            self.f1_score = 2 * ((self.accuracy_score * self.sensibility) /
                                 (self.accuracy_score + self.sensibility))

    def calculate_roc(self, probability_positive_class, y_test):
        fpr, tpr, thresholds = roc_curve(y_test, probability_positive_class)
        self.area_under_curve = roc_auc_score(
            y_test, probability_positive_class)
        self.receiver_operating_characteristic = {
            "fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    def print_results(self):
        print("Results:")
        if self.confusion_matrixes.any():
            print("confusion_matrix => ")
            print(self.confusion_matrixes)
        if self.accuracy_score:
            print("accuracy_score => ")
            print(self.accuracy_score)
        if self.sensibility:
            print("sensibility => ")
            print(self.sensibility)
        if self.specificity:
            print("specificity => ")
            print(self.specificity)
        if self.value_predictive_positive:
            print("value_predictive_positive => ")
            print(self.value_predictive_positive)
        if self.value_predictive_negative:
            print("value_predictive_negative => ")
            print(self.value_predictive_negative)
        if self.f1_score:
            print("f1_score => ")
            print(self.f1_score)
        if self.area_under_curve:
            print("area_under_curve => ")
            print(self.area_under_curve)
        if self.receiver_operating_characteristic:
            plt.figure(figsize=(8, 6))
            plt.plot(
                self.receiver_operating_characteristic.fpr,
                self.receiver_operating_characteristic.tpr,
                color='blue',
                lw=2,
                label='ROC curve (area = {:.2f})'.format(self.area_under_curve)
            )
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()
