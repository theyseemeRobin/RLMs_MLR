from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report


class ClassificationMetrics:
    """
    A wrapper for true label and predicted label arrays that allows for easy retrieval of common classification metrics.
    """
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.true_labels, self.predicted_labels)

    @property
    def accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    @property
    def precision(self):
        return precision_score(self.true_labels, self.predicted_labels, average='weighted')

    @property
    def f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')

    @property
    def classification_report(self):
        return classification_report(self.true_labels, self.predicted_labels)