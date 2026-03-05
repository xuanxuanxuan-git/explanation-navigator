import json
import threading

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import plotly.graph_objects as go
import plotly.io as pio


# ----------------------------
# Global cached state
# ----------------------------
_STATE_LOCK = threading.Lock()
_STATE = {
    "ready": False,
    "X_train": None,
    "X_test": None,               # pd.DataFrame
    "y_train": None,
    "y_test": None,
    "scaler": None,
    "model": None,
    "X_test_scaled": None,        # np.ndarray
    "explainer": None,
    "shap_values": None,          # np.ndarray (n_test, n_features)
    "expected_value": None,       # float
}

_OP_MAP = {
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "==": "==",
    "gt": ">",
    "ge": ">=",
    "lt": "<",
    "le": "<=",
    "eq": "==",
}

def _init_if_needed(n_points=1000, test_size=0.2, random_state=42):
    if _STATE["ready"]:
        return

    with _STATE_LOCK:
        if _STATE["ready"]:
            return

        X, y = shap.datasets.california(n_points=n_points)
        X = pd.DataFrame(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        # expected_value for regression is typically a scalar baseline
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.array(expected_value).reshape(-1)[0])

        _STATE.update({
            "X_train": X_train.reset_index(drop=True),
            "X_test": X_test.reset_index(drop=True),
            "y_train": np.array(y_train),
            "y_test": np.array(y_test),
            "scaler": scaler,
            "model": model,
            "X_test_scaled": np.array(X_test_scaled),
            "explainer": explainer,
            "shap_values": np.array(shap_values),
            "expected_value": float(expected_value),
            "ready": True,
        })


def _plotly_payload(fig, *, display_mode_bar=False, meta=None):
    fig_json = json.loads(pio.to_json(fig))
    return {
        "type": "plotly",
        "figure": fig_json,
        "config": {"displayModeBar": display_mode_bar, "responsive": True},
        "meta": meta or {},
    }

def _check_instance_id(instance_id: int):
    X_test = _STATE["X_test"]
    n = len(X_test)
    if instance_id < 0 or instance_id >= n:
        raise ValueError(f"Invalid instance_id={instance_id}. Must be between 0 and {n-1}.")

def _local_rows(instance_id: int):
    X_test = _STATE["X_test"]
    shap_values = _STATE["shap_values"]
    feature_names = X_test.columns.tolist()

    vals = shap_values[instance_id].astype(float)
    rows = [{"feature": f, "shap_value": float(v)} for f, v in zip(feature_names, vals)]
    return rows

def _maybe_json_loads(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return x
    return x

# ----------------------------
# Tools
# ----------------------------

def generate_shap_bar_plot(instance_id: int, max_display: int = 10):
    _init_if_needed()
    instance_id = int(instance_id)
    max_display = int(max_display)

    _check_instance_id(instance_id)

    rows = _local_rows(instance_id)
    rows_sorted = sorted(rows, key=lambda r: abs(r["shap_value"]), reverse=True)[:max_display]
    # round to 4 dp for returned data
    rows_sorted = [
        {"feature": r["feature"], "shap_value": round(float(r["shap_value"]), 4)}
        for r in rows_sorted
    ]
    # For horizontal bars: feature names on Y, SHAP values on X
    features = [r["feature"] for r in rows_sorted][::-1]
    values = [r["shap_value"] for r in rows_sorted][::-1]
    colors = ["#ef4444" if v < 0 else "#3b82f6" for v in values]

    fig = go.Figure(
        data=[go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker={"color": colors},
            customdata=features,
            hovertemplate="Feature: %{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )],
        layout=go.Layout(
            title=f"Local feature attribution for instance {instance_id}",
            xaxis={"title": "SHAP value"},
            yaxis={"title": "Feature"},
            margin={"l": 140, "r": 20, "t": 55, "b": 40},
        ),
    )

    return {
        "data": rows_sorted,
        "visualisation": _plotly_payload(fig, display_mode_bar=False, meta={"tool": "generate_shap_bar_plot", "instance_id": instance_id}),
    }


def generate_shap_summary_plot(max_display: int = 10):
    """
    SHAP-like global summary.
    """
    _init_if_needed()
    max_display = int(max_display)

    X_test = _STATE["X_test"]
    shap_values = _STATE["shap_values"]

    feature_names = X_test.columns.tolist()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    rows = [{"feature": f, "mean_abs_shap": float(v)} for f, v in zip(feature_names, mean_abs)]
    rows_sorted = sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)[:max_display]
    rows_sorted = [
    {"feature": r["feature"], "mean_abs_shap": round(float(r["mean_abs_shap"]), 4)} for r in rows_sorted
    ]

    y = [r["feature"] for r in rows_sorted][::-1]
    x = [r["mean_abs_shap"] for r in rows_sorted][::-1]

    fig = go.Figure(
        data=[go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker={"color": ["#10b981"] * len(x)},
            customdata=y,
            hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )],
        layout=go.Layout(
            title="Global feature importance",
            xaxis={"title": "Mean |SHAP|"},
            yaxis={"title": "Feature"},
            margin={"l": 120, "r": 20, "t": 55, "b": 40},
        ),
    )

    return {
        "data": rows_sorted,
        "visualisation": _plotly_payload(fig, display_mode_bar=False, meta={"tool": "generate_shap_summary_plot", "kind": "bar"}),
    }


def get_feature_attribution_ranking(instance_id: int):
    """
    Returns local ranking sorted by abs(shap_value) desc.
    """
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    rows = _local_rows(instance_id)
    rows_sorted = sorted(rows, key=lambda r: abs(r["shap_value"]), reverse=True)

    return {"data": rows_sorted, "visualisation": None}


def get_individual_prediction(instance_id: int):
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    model = _STATE["model"]
    scaler = _STATE["scaler"]
    X_test = _STATE["X_test"]

    x = X_test.iloc[[instance_id]]
    pred = float(model.predict(scaler.transform(x))[0])

    return {
        "data": {"instance_id": instance_id, "probability": round(pred, 2)},
        "visualisation": None,
    }


def get_average_prediction(source: str = "all", indices=None):
    """
    source: "all" or "indices"
    indices: optional list[int] (e.g. from get_subgroup)
    """
    _init_if_needed()
    model = _STATE["model"]
    scaler = _STATE["scaler"]
    X_test = _STATE["X_test"]

    source = (source or "test").strip().lower()

    if source == "all":
        X = X_test
    elif source == "indices":
        if not isinstance(indices, list) or len(indices) == 0:
            return {"data": "indices must be a non-empty list when source='indices'.", "visualisation": None}
        # validate and slice
        idx = [int(i) for i in indices if 0 <= int(i) < len(X_test)]
        if not idx:
            return {"data": "No valid indices provided.", "visualisation": None}
        X = X_test.iloc[idx]
    else:
        return {"data": f"Unknown source='{source}'. Use 'test' or 'indices'.", "visualisation": None}

    preds = model.predict(scaler.transform(X)).astype(float)
    return {
        "data": {
            "source": source,
            "count": int(len(X)),
            "average_prediction": float(np.mean(preds)),
        },
        "visualisation": None,
    }


def get_cp_plot(instance_id: int, feature: str, grid_points: int = 100):
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    feature = (feature or "").strip()
    grid_points = int(grid_points)

    X_test = _STATE["X_test"]
    scaler = _STATE["scaler"]
    model = _STATE["model"]

    if feature not in X_test.columns:
        return {"data": f"Unknown feature '{feature}'.", "visualisation": None}

    x0 = X_test.iloc[[instance_id]].copy()
    base_val = float(x0[feature].iloc[0])

    col = X_test[feature].astype(float)
    grid = np.linspace(float(col.min()), float(col.max()), grid_points)

    preds = []
    for v in grid:
        xv = x0.copy()
        xv[feature] = v
        preds.append(float(model.predict(scaler.transform(xv))[0]))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=grid.tolist(),
                y=preds,
                mode="lines",
                line={"color": "#6366f1"},
                hovertemplate=f"{feature}: %{{x:.4f}}<br>Prediction: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
            go.Scatter(
                x=[base_val],
                y=[float(model.predict(scaler.transform(x0))[0])],
                mode="markers",
                marker={"size": 10, "color": "#ef4444"},
                hovertemplate=f"Current {feature}: %{{x:.4f}}<br>Prediction: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
        ],
        layout=go.Layout(
            title=f"CP plot — instance {instance_id}, feature: {feature}",
            xaxis={"title": feature},
            yaxis={"title": "Model prediction"},
            margin={"l": 60, "r": 20, "t": 55, "b": 40},
        ),
    )

    return {
        "data": {
            "instance_id": instance_id,
            "feature": feature,
            "base_value": base_val,
            "grid": [round(v, 4) for v in grid.tolist()],
            "prediction": [round(p, 4) for p in preds],
        },
        "visualisation": _plotly_payload(fig, display_mode_bar=False, meta={"tool": "get_cp_plot", "instance_id": instance_id, "feature": feature}),
    }

def get_counterfactual_explanation():
    # retrieve counterfactual explanations
    return



def get_subgroup(filters: dict, limit: int = 500):
    _init_if_needed()
    X_test = _STATE["X_test"]
    limit = int(limit)

    # If filters accidentally comes as a JSON string, parse it
    filters = _maybe_json_loads(filters)
    if not isinstance(filters, dict):
        return {"data": {"size": 0, "indices": [], "filters": filters}, "visualisation": None}

    df = X_test.copy()
    for feat, cond in filters.items():
        if feat not in df.columns:
            continue
        if not isinstance(cond, dict):
            continue

        # Support two styles:
        #  A) {"op": ">", "value": 1}
        #  B) {"gt": 1} or {"le": 3}
        if "op" in cond:
            op = _OP_MAP.get(cond.get("op"))
            val = cond.get("value")
            if op is None or val is None:
                continue
        else:
            # pick first recognized operator key
            op = None
            val = None
            for k, v in cond.items():
                if k in _OP_MAP:
                    op = _OP_MAP[k]
                    val = v
                    break
            if op is None:
                continue

        if op == ">":
            df = df[df[feat] > val]
        elif op == ">=":
            df = df[df[feat] >= val]
        elif op == "<":
            df = df[df[feat] < val]
        elif op == "<=":
            df = df[df[feat] <= val]
        elif op == "==":
            df = df[df[feat] == val]

    indices = df.index.astype(int).tolist()
    return {
        "data": {
            "size": int(len(df)),
            "indices": indices,
            "filters": filters,
        },
        "visualisation": None,
    }



def predict_with_feature_changes(instance_id: int, changes: dict):
    """
    changes example:
      {"MedInc": 4.2, "AveRooms": 6.0}
    """
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    if not isinstance(changes, dict) or not changes:
        return {"data": "changes must be a non-empty object of feature -> new_value.", "visualisation": None}

    model = _STATE["model"]
    scaler = _STATE["scaler"]
    X_test = _STATE["X_test"]

    x0 = X_test.iloc[[instance_id]].copy()
    pred0 = float(model.predict(scaler.transform(x0))[0])

    x1 = x0.copy()
    for feat, val in changes.items():
        if feat not in x1.columns:
            return {"data": f"Unknown feature '{feat}'.", "visualisation": None}
        x1[feat] = float(val)

    pred1 = float(model.predict(scaler.transform(x1))[0])

    return {
        "data": {
            "instance_id": instance_id,
            "original_prediction": pred0,
            "new_prediction": pred1,
            "change_in_prediction": float(pred1 - pred0),
            "changes": {k: float(v) for k, v in changes.items()},
            "original_values": {k: float(x0[k].iloc[0]) for k in changes.keys()},
            "new_values": {k: float(x1[k].iloc[0]) for k in changes.keys()},
        },
        "visualisation": None,
    }


def resolve_feature_name():
    pass


available_tools_mapping = {
    "generate_local_shap_bar_plot": generate_shap_bar_plot,
    "generate_global_shap_summary_plot": generate_shap_summary_plot,
    # "get_feature_attribution_ranking": get_feature_attribution_ranking,
    "get_individual_prediction": get_individual_prediction,
    "get_average_prediction": get_average_prediction,
    "get_cp_plot": get_cp_plot,
    "get_counterfactual_explanation": get_counterfactual_explanation,
    "get_subgroup": get_subgroup,
    "predict_with_feature_changes": predict_with_feature_changes,
}
        