import numpy as np
import numpy.typing as npt

from common import RegContext, EVAL_FUNCTIONS, Predictor


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
    predictor = train_model_with_dropout(
        dropout_prob=dropout_prob,
        X=X_train,
        Y=Y_train,
        reg_context=reg_context,
    )
    return EVAL_FUNCTIONS[reg_context.eval_criterion](
        yhat=predictor(X_test),
        y=Y_test,
    )


def train_model_with_dropout(
        dropout_prob: float,
        X: npt.NDArray,
        Y: npt.NDArray,
        reg_context: RegContext,
) -> Predictor:
    assert dropout_prob >= 0
    assert dropout_prob < 1
    # Create lists of `mc_replicates` datasets with dropout
    X_train_all = []
    Y_train_all = []
    for _ in range(reg_context.M):
        # This is a matrix whose component (i, j) is the proposition that
        #   entry (i, j) of X should be dropped (i.e., set to 0)
        dropout_mask = np.random.random(X.shape) < dropout_prob
        X_with_dropout = (dropout_mask * X) / (1 - dropout_prob)
        X_train_all.append(X_with_dropout)
        Y_train_all.append(Y)
    
    # Concatenate to form a single large training dataset and label set
    X_train_full = np.concatenate(X_train_all, axis=0)
    Y_train_full = np.concatenate(Y_train_all, axis=0)

    return reg_context.learning_alg(X_train_full, Y_train_full)
