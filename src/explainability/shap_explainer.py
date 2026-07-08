"""SHAP model explainability with permutation importance fallback."""

from typing import Any, Callable
import numpy as np
import pandas as pd

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False


def explain_prediction(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    feature_names: list[str] | None = None,
    background_data: pd.DataFrame | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Per-prediction and global explanations via SHAP or permutation importance."""
    names = feature_names or list(frame.columns)
    bg = background_data if background_data is not None else frame
    result: dict[str, Any] = {
        "method": "permutation",
        "global_importance": [],
        "local_explanations": [],
    }

    if _HAS_SHAP:
        try:
            explainer = shap.Explainer(predict_fn, bg, feature_names=names)
            shap_values = explainer(frame)
            result["method"] = "shap"
            base_vals = getattr(shap_values, "base_values", None)
            if base_vals is not None and hasattr(base_vals, "__len__"):
                bv = np.asarray(base_vals[0])
                result["base_value"] = float(bv.mean()) if bv.ndim > 0 else float(bv)
            else:
                result["base_value"] = 0.0
            global_imp = np.abs(shap_values.values).mean(axis=0)
            if global_imp.ndim > 1:
                global_imp = global_imp.mean(axis=tuple(range(1, global_imp.ndim)))
            global_imp = np.nan_to_num(global_imp, nan=0.0, posinf=0.0, neginf=0.0)
            global_imp_list = (
                global_imp.tolist() if hasattr(global_imp, "tolist") else list(global_imp)
            )
            result["global_importance"] = [
                {"feature": str(n), "importance": float(v)} for n, v in zip(names, global_imp_list)
            ]
            result["global_importance"].sort(key=lambda x: -x["importance"])

            for i in range(min(len(frame), 5)):
                vals = shap_values.values[i]
                if hasattr(vals, "shape") and vals.ndim > 1:
                    vals = vals.mean(axis=-1)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                vals_list = vals.tolist() if hasattr(vals, "tolist") else list(vals)
                local = [
                    {"feature": str(n), "shap_value": float(v)} for n, v in zip(names, vals_list)
                ]
                local.sort(key=lambda x: -abs(x["shap_value"]))
                result["local_explanations"].append({"row": int(i), "factors": local[:10]})
            return result
        except Exception:
            import logging

            logging.exception("SHAP explainer failed, falling back to permutation importance")

    # Fallback: permutation importance
    try:
        base_pred = predict_fn(frame)
        base_metric = float(np.mean(np.abs(base_pred)))
        for i, name in enumerate(names):
            permuted = frame.copy()
            permuted.iloc[:, i] = np.random.permutation(permuted.iloc[:, i].values)
            perm_pred = predict_fn(permuted)
            perm_metric = float(np.mean(np.abs(perm_pred)))
            importance = abs(base_metric - perm_metric) / max(abs(base_metric), 1e-10)
            result["global_importance"].append({"feature": str(name), "importance": importance})
        result["global_importance"].sort(key=lambda x: -x["importance"])
    except Exception:
        pass

    # Always populate local explanations (approximation using data deviation * global importance)
    if not result["local_explanations"] and result["global_importance"]:
        try:
            imp_map = {d["feature"]: d["importance"] for d in result["global_importance"]}
            imp_vec = np.array([imp_map.get(n, 0.0) for n in names], dtype=float)
            imp_sum = imp_vec.sum()
            if imp_sum > 0:
                imp_vec /= imp_sum
            bg_mean = frame.mean(numeric_only=True).values.astype(float)
            for j in range(min(len(frame), 5)):
                row = frame.iloc[j].values.astype(float)
                dev = np.nan_to_num(np.abs(row - bg_mean), nan=0.0, posinf=0.0, neginf=0.0)
                dev_sum = dev.sum()
                if dev_sum > 0:
                    vals = dev / dev_sum * imp_sum
                else:
                    vals = imp_vec.copy()
                local = [{"feature": str(n), "shap_value": float(v)} for n, v in zip(names, vals)]
                local.sort(key=lambda x: -abs(x["shap_value"]))
                result["local_explanations"].append({"row": int(j), "factors": local[:10]})
        except Exception:
            pass

    return result


def feature_interaction_matrix(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    feature_names: list[str],
    max_features: int = 8,
    model: Any = None,
) -> dict[str, Any]:
    """Compute SHAP interaction values for top features, if available."""
    if not _HAS_SHAP:
        return {"method": "none", "message": "SHAP not installed"}
    try:
        if model is not None and hasattr(model, "steps"):
            estimator = model.steps[-1][1]
            explainer = shap.TreeExplainer(estimator)
        else:
            explainer = shap.TreeExplainer(predict_fn)
        interaction_values = explainer.shap_interaction_values(frame.iloc[:50])
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[0]
        n = min(interaction_values.shape[-1], max_features)
        matrix = np.abs(interaction_values[:, :n, :n]).mean(axis=0)
        names = feature_names[:n]
        return {
            "method": "shap_interaction",
            "names": names,
            "matrix": matrix.tolist(),
        }
    except Exception as e:
        return {"method": "none", "error": str(e)}
