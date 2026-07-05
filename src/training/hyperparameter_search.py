"""Model selection with hyperparameter tuning."""

from itertools import product
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src.dataset.loader import TARGETS
from src.surrogate.train import create_model


def select_model(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[object, dict[str, float]]:
    """Grid-search over model kinds and n_estimators, return best model and its config."""
    kinds = ["extra_trees", "random_forest", "hist_gradient_boosting", "mlp"]
    estimator_options = [200, 400]
    best_rmse = float("inf")
    best_model = None
    best_config = {}
    for kind, n_est in product(kinds, estimator_options):
        try:
            model = create_model(kind, n_estimators=n_est).fit(train)
            pred = model.predict(validation)
            rmse = float(root_mean_squared_error(validation[TARGETS], pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_config = {"kind": kind, "n_estimators": n_est, "rmse": rmse}
        except Exception:
            continue
    return best_model, best_config
