import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# --- Base Metric Class ---

class Metric:
    """
    An abstract base class for a metric.

    This class provides a common interface for all metric calculations.
    It handles the conversion of PyTorch tensors to NumPy arrays, which is
    required by scikit-learn's metric functions. Subclasses should
    implement the `_compute` method for the actual metric logic.
    """
    def __call__(self, y: torch.Tensor, yhat: torch.Tensor) -> float:
        """
        Calculates the metric.

        Args:
            y (torch.Tensor): The ground truth labels.
            yhat (torch.Tensor): The model's predictions (logits or probabilities).

        Returns:
            float: The calculated metric value.
        """
        # Detach tensors from the computation graph and move them to the CPU.
        y_ = y.detach().cpu().numpy()
        yhat_ = yhat.detach().cpu().numpy()

        # Call the specific computation method implemented by the subclass.
        return self._compute(y_, yhat_)

    def _compute(self, y: np.ndarray, yhat: np.ndarray) -> float:
        raise NotImplementedError()

# --- Specific Metric Implementations ---

class Accuracy(Metric):
    """Calculates the accuracy score."""

    def __repr__(self) -> str:
        """Provides a string representation for the metric, e.g., for logging."""
        return "Accuracy"

    def _compute(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """
        Computes accuracy.

        Args:
            y (np.ndarray): The ground truth labels.
            yhat (np.ndarray): The model's predictions.

        Returns:
            float: The accuracy score.
        """
        # Get the index of the highest value in the prediction array (the predicted class).
        predicted_classes = yhat.argmax(axis=1)
        # Compare predicted classes to true labels and calculate the mean.
        return (predicted_classes == y).sum() / len(yhat)


class F1Score(Metric):
    """Calculates the F1 score, the harmonic mean of precision and recall."""

    def __init__(self, average: str = 'weighted'):
        """
        Args:
            average (str): The averaging strategy for multi-class problems.
                'micro': Global calculation.
                'macro': Unweighted mean per class.
                'weighted': Mean per class, weighted by support (number of true instances).
        """
        self.average = average

    def __repr__(self) -> str:
        """Provides a string representation for the metric."""
        return f"F1Score_{self.average}"

    def _compute(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """Computes the F1 score using scikit-learn."""
        return f1_score(
            y, yhat.argmax(axis=1), average=self.average, zero_division=np.nan
        )


class Recall(Metric):
    """Calculates recall (sensitivity), the ability to find all positive samples."""

    def __init__(self, average: str = 'weighted'):
        """
        Args:
            average (str): The averaging strategy for multi-class problems.
        """
        self.average = average

    def __repr__(self) -> str:
        """Provides a string representation for the metric."""
        return f"Recall_{self.average}"

    def _compute(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """Computes the recall score using scikit-learn."""
        return recall_score(
            y, yhat.argmax(axis=1), average=self.average, zero_division=np.nan
        )


class Precision(Metric):
    """Calculates precision, the ability to not label a negative sample as positive."""

    def __init__(self, average: str = 'weighted'):
        """
        Args:
            average (str): The averaging strategy for multi-class problems.
        """
        self.average = average

    def __repr__(self) -> str:
        """Provides a string representation for the metric."""
        return f"Precision_{self.average}"

    def _compute(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """Computes the precision score using scikit-learn."""
        return precision_score(
            y, yhat.argmax(axis=1), average=self.average, zero_division=np.nan
        )