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
    assert dropout_prob < 1
    # TODO: Jordan combines the data across replicates into
    #   one large training set and one large test set. I haven't ported this over
    #   quite yet, because it bloats the code a bit (both in size and complexity),
    #   and I don't understand why it's doing that yet.
    #   We should do more research/ask about it on Slack.
    mc_replicates = reg_context.M
    evaluations_by_replicate = np.zeros(mc_replicates)
    for r in range(mc_replicates):
        predictor_with_dropout = train_model_with_dropout(
            dropout_prob=dropout_prob,
            learning_alg=reg_context.learning_alg,
            X=X_train,
            Y=Y_train,
        )
        Y_hat_test = predictor_with_dropout(X_test)
        evaluations_by_replicate[r] = EVAL_FUNCTIONS[reg_context.eval_criterion](
            Y_hat_test,
            Y_test,
        )
    return np.mean(evaluations_by_replicate)


# TODO: Verify that this is generating the final model correctly.
#   In particular, should it use the combined dataset with all
#   M replicates?
def train_model_with_dropout(
        dropout_prob: float,
        learning_alg: LearningAlg,
        X: npt.NDArray,
        Y: npt.NDArray,
) -> Predictor:
    # This is a matrix whose component (i, j) is the proposition that
    #   entry (i, j) of X should be dropped (i.e., set to 0)
    dropout_mask = np.random.random(X.shape) < dropout_prob
    X_with_dropout = (dropout_mask * X) / (1 - dropout_prob)
    return learning_alg(X_with_dropout, Y)