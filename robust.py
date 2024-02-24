import typing as t

import numpy.typing as npt

from common import EVAL_FUNCTIONS, Predictor, RegContext


def evaluate_c_on_split(
        c: npt.NDArray,
        X_train: npt.NDArray,
        X_test: npt.NDArray,
        Y_train: npt.NDArray,
        Y_test: npt.NDArray,
        reg_context: RegContext,
) -> float:
    predictor = train_model_with_robust(
        c=c,
        reg_context=reg_context,
        X=X_train,
        Y=Y_train,
    )
    Y_hat_test = predictor(X_test)
    return EVAL_FUNCTIONS[reg_context.eval_criterion](
        Y_hat_test,
        Y_test,
    )


def train_model_with_robust(
        c: npt.NDArray,
        X: npt.NDArray,
        Y: npt.NDArray,
        reg_context: RegContext,
) -> Predictor:
    perturbation_matrices = generate_candidate_perturbations(c)
    models: t.List[Predictor] = [
        reg_context.learning_alg(X + delta, Y)
        for delta in perturbation_matrices
    ]
    # also consider the model that is optimal under the unperturbed data
    models.append(reg_context.learning_alg(X, Y))

    # evaluate the loss of each model under each perturbation matrix.
    # compute the maximum loss across perturbation matrices for each model.
    maximum_loss_by_model = [
        max(
            reg_context.eval_criterion(
                yhat=model(X + pm),
                y=Y,
            )
            for pm in perturbation_matrices
        )
        for model in models
    ]
    # return model that has the lowest maximum loss
    return models[min(range(len(models)), key=maximum_loss_by_model.__getitem__)]
    

def generate_candidate_perturbations(c: npt.NDArray, num: int=100) -> t.List[npt.NDArray]:
    # TODO: Implement this.
    pass