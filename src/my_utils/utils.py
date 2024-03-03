import os
import shutil
import tensorboard.program
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report
from tensorboard import program
import sklearn


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


def copy_file(source_file, destination_dir):
    """
    Copies a file from the current location to the desired destination. Creates the destination directory if it does
    not yet exist.

    Parameters
    ----------
    source_file : str
        Current location of the file.
    destination_dir : str
        Destination location of the file.
    """

    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_file))
    shutil.copy(source_file, destination_path)


def open_tensorboard(log_dir):
    """
    Parameters
    ----------
    log_dir : str
    """
    tb = program.TensorBoard()
    os.makedirs(log_dir, exist_ok=True)
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")