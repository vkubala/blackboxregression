import numpy as np
import numpy.typing as npt

from common import RegContext, EVAL_FUNCTIONS, LearningAlg, Predictor


def evaluate_dropout_probability_on_split(
        dropout_prob: float,
        X_train: npt.NDArray,
        X_test: npt.NDArray,
        Y_train: npt.NDArray,
        Y_test: npt.NDArray,
        reg_context: RegContext,
) -> float:
    """Compute the average loss (according to the evlauation criterion)
    achived by the given dropout probability on the given folds of data."""
    pass


def train_model_with_dropout(
        dropout_prob: float,
        X: npt.NDArray,
        Y: npt.NDArray,
        reg_context: RegContext,
) -> Predictor:
    # Create lists of `mc_replicates` datasets with dropout
    



    # This is a matrix whose component (i, j) is the proposition that
    #   entry (i, j) of X should be dropped (i.e., set to 0)
    # dropout_mask = np.random.random(X.shape) < dropout_prob
    # X_with_dropout = (dropout_mask * X) / (1 - dropout_prob)
    # return learning_alg(X_with_dropout, Y)