import json
import threading

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

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

def _init_if_needed(test_size=0.2, random_state=42):
    if _STATE["ready"]:
        return

    with _STATE_LOCK:
        if _STATE["ready"]:
            return
        df = pd.read_csv("use_case_data/heloc_dataset.csv")  
        # Convert special missing codes to NaN first
        df = df.replace([-9, -8, -7], np.nan)

        # Remove rows where non-target feature values are missing
        feature_cols = [c for c in df.columns if c != "RiskPerformance"]
        df = df.dropna(subset=feature_cols, how="any").reset_index(drop=True)

        # Target: RiskPerformance (Good/Bad)
        # predicting the risk of being a bad borrower
        y = df["RiskPerformance"].map({"Good": 0, "Bad": 1}).values
        X = df.drop(columns=["RiskPerformance"]).copy()

        # Fill remaining missing values with column medians
        # X = X.fillna(X.median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        # base_model = RandomForestClassifier(
        #     n_estimators=200,
        #     random_state=random_state,
        #     n_jobs=-1
        # )
        # base_model.fit(X_train_scaled, y_train)

        # importances = base_model.feature_importances_
        # feature_names = X.columns

        # top_idx = np.argsort(importances)[::-1][:10]
        # top_features = feature_names[top_idx]
        # print(top_features) 
        top_features = ['ExternalRiskEstimate', 'NetFractionRevolvingBurden', 'AverageMInFile', 'MSinceOldestTradeOpen', 'MSinceMostRecentDelq', 'PercentTradesNeverDelq', 'NetFractionInstallBurden', 'PercentTradesWBalance', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days']
        
        # Reduce dataset
        X_train = X_train[top_features].copy()
        X_test = X_test[top_features].copy()

        # Re-scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        # For binary classification -> take class 1 (bad borrower)
        shap_values = shap_values[:, :, 1]

        # expected_value = explainer.expected_value
        # if isinstance(expected_value, list):
        #     expected_value = expected_value[1]

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
            # "expected_value": float(expected_value),
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

def _euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

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
        "visualisation": _plotly_payload(fig, meta={"tool": "generate_shap_bar_plot", "instance_id": instance_id}),
    }

def generate_shap_summary_plot(source: str = "all", indices=None, max_display: int = 10):
    """
    Generate a SHAP summary plot for either the whole dataset or a subset of instances.

    source: "all" or "indices"
    indices: optional list[int] (e.g. returned by get_subgroup)
    """
    _init_if_needed()

    source = (source or "all").strip().lower()
    max_display = int(max_display)

    X_test = _STATE["X_test"]
    shap_values = _STATE["shap_values"]

    if source == "all":
        X = X_test
        shap_subset = shap_values

    elif source == "indices":
        if not isinstance(indices, list) or len(indices) == 0:
            return {"data": "indices must be a non-empty list when source='indices'.", "visualisation": None}

        idx = [int(i) for i in indices if 0 <= int(i) < len(X_test)]
        if not idx:
            return {"data": "No valid indices provided.", "visualisation": None}

        X = X_test.iloc[idx]
        shap_subset = shap_values[idx]

    else:
        return {"data": f"Unknown source='{source}'. Use 'all' or 'indices'.", "visualisation": None}

    feature_names = X_test.columns.tolist()

    mean_abs = np.mean(np.abs(shap_subset), axis=0)

    rows = [{"feature": f, "mean_abs_shap": float(v)} for f, v in zip(feature_names, mean_abs)]
    rows_sorted = sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)[:max_display]
    rows_sorted = [{"feature": r["feature"], "mean_abs_shap": round(float(r["mean_abs_shap"]), 4)} for r in rows_sorted]

    y = [r["feature"] for r in rows_sorted][::-1]
    x = [r["mean_abs_shap"] for r in rows_sorted][::-1]

    title = "Global feature importance" if source == "all" else "Feature importance for selected subgroup"

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
            title=title,
            xaxis={"title": "Mean |SHAP|"},
            yaxis={"title": "Feature"},
            margin={"l": 120, "r": 20, "t": 55, "b": 40},
        ),
    )

    return {
        "data": {
            "source": source,
            "count": int(len(X)),
            "features": rows_sorted,
        },
        "visualisation": _plotly_payload(fig, display_mode_bar=False, meta={"tool": "generate_shap_summary_plot"}),
    }


def get_instance_features_and_prediction(instance_id: int):
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    model = _STATE["model"]
    X_test = _STATE["X_test"]
    X_scaled = _STATE["X_test_scaled"]

    x = X_scaled[[instance_id]]
    # pred = float(model.predict(x)[0])
    pred = float(model.predict_proba(x)[0][1])

    return {
        "data": {"instance_id": instance_id,
            "instance_features": X_test.iloc[instance_id].to_dict(),
            "prediction": round(pred, 4)},
        "visualisation": None,
    }


def get_average_prediction(source: str = "all", indices=None):
    """
    source: "all" or "indices"
    indices: optional list[int] (e.g. from get_subgroup)
    """
    _init_if_needed()
    model = _STATE["model"]
    X_scaled = _STATE["X_test_scaled"]

    source = (source or "test").strip().lower()

    if source == "all":
        X = X_scaled 
    elif source == "indices":
        if not isinstance(indices, list) or len(indices) == 0:
            return {"data": "indices must be a non-empty list when source='indices'.", "visualisation": None}
        # validate and slice
        idx = [int(i) for i in set(indices) if 0 <= int(i) < len(X_scaled)]
        if not idx:
            return {"data": "No valid indices provided.", "visualisation": None}
        X = X_scaled[idx]
    else:
        return {"data": f"Unknown source='{source}'. Use 'test' or 'indices'.", "visualisation": None}

    preds = model.predict_proba(X)[:, 1]
    avg = float(np.mean(preds))
    return {
        "data": {
            "source": source,
            "count": int(len(X)),
            "average_probability_of_default": round(avg, 4),
        },
        "visualisation": None,
    }

# Change in probability of predicting as bad borrower
def get_cp_plot(instance_id: int, feature: str, grid_points: int = 150):
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
        preds.append(float(model.predict_proba(scaler.transform(xv))[0][1]))
    base_pred = float(model.predict_proba(scaler.transform(x0))[0][1])
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
                y=[base_pred],
                mode="markers",
                marker={"size": 10, "color": "#ef4444"},
                hovertemplate=f"Current {feature}: %{{x:.4f}}<br>Prediction: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
        ],
        layout=go.Layout(
            title=f"CP plot -- instance {instance_id}, feature: {feature}",
            xaxis={"title": feature},
            yaxis={"title": "Probability of default"},
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


def get_partial_dependence_plot(feature: str, grid_points: int = 150):
    """
    Generate Partial Dependence Plot (PDP) for a feature.

    PDP shows the average model prediction as the feature varies,
    marginalising over all other features.
    """
    _init_if_needed()

    feature = (feature or "").strip()
    grid_points = int(grid_points)

    X_test = _STATE["X_test"]
    X_scaled = _STATE["X_test_scaled"]
    model = _STATE["model"]
    scaler = _STATE["scaler"]

    if feature not in X_test.columns:
        return {"data": f"Unknown feature '{feature}'.", "visualisation": None}

    feature_idx = X_test.columns.get_loc(feature)

    col = X_test[feature].astype(float)
    grid = np.linspace(col.min(), col.max(), grid_points)

    # --------- SCALE GRID CORRECTLY ----------
    col_mean = scaler.mean_[feature_idx]
    col_scale = scaler.scale_[feature_idx]
    grid_scaled = (grid - col_mean) / col_scale

    # --------- VECTORIZED PDP ----------
    n = X_scaled.shape[0]

    # Repeat dataset for each grid value
    X_rep = np.repeat(X_scaled, grid_points, axis=0)

    # Tile grid values
    grid_tiled = np.tile(grid_scaled, n)

    # Replace feature column
    X_rep[:, feature_idx] = grid_tiled

    preds = model.predict_proba(X_rep)[:, 1]
    preds = preds.reshape(n, grid_points)
    pdp_values = preds.mean(axis=0)

    # ---------- PLOT ----------
    fig = go.Figure()

    # PDP line
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=pdp_values,
            mode="lines",
            line={"color": "#2563eb"},
            name="PDP",
            hovertemplate=f"{feature}: %{{x:.2f}}<br>Avg Prediction: %{{y:.4f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Partial Dependence Plot — {feature}",
        xaxis={"title": feature},
        yaxis={"title": "Average prediction"},
        margin={"l": 60, "r": 60, "t": 55, "b": 40},
    )

    return {
        "data": {
            "feature": feature,
            "grid": [round(v, 4) for v in grid.tolist()],
            "average_prediction": [round(p, 4) for p in pdp_values.tolist()],
        },
        "visualisation": _plotly_payload(
            fig,
            display_mode_bar=False,
            meta={
                "tool": "get_partial_dependence_plot",
                "feature": feature,
            },
        ),
    }

def get_counterfactual_explanation(instance_id: int, target: float = None, max_steps: int = 50):
    """
    Generate a counterfactual explanation for an instance.

    Strategy:
    - Greedy feature-wise search
    - At each step, modify the best feature that moves prediction toward target
    """

    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    X_test = _STATE["X_test"]
    X_train = _STATE["X_train"]
    scaler = _STATE["scaler"]
    model = _STATE["model"]

    x0 = X_test.iloc[[instance_id]].copy()
    x_cf = x0.copy()

    original_pred = float(model.predict_proba(scaler.transform(x0))[0][1])

    # Default target: decrease the probability by 20%
    # TODO: fix
    if target is None:
        return {
            "error": "Target probability is required. Please specify a desired probability."
        }

    feature_names = X_test.columns.tolist()

    # Precompute feature ranges
    feature_ranges = {
        f: (float(X_train[f].min()), float(X_train[f].max()))
        for f in feature_names
    }

    current_pred = original_pred

    # ---------- GREEDY SEARCH ----------
    for _ in range(max_steps):

        best_feature = None
        best_value = None
        best_pred = current_pred

        for f in feature_names:
            min_v, max_v = feature_ranges[f]

            # try small steps in both directions
            candidates = np.linspace(min_v, max_v, 20)

            for v in candidates:
                temp = x_cf.copy()
                temp[f] = v

                pred = float(model.predict_proba(scaler.transform(temp))[0][1])

                # move closer to target
                if abs(pred - target) < abs(best_pred - target):
                    best_pred = pred
                    best_feature = f
                    best_value = v

        # no improvement -> stop
        if best_feature is None:
            break

        # apply best change
        x_cf[best_feature] = best_value
        current_pred = best_pred

        # stop early if close enough
        if abs(current_pred - target) < 1e-3:
            break

    # ---------- EXTRACT CHANGES ----------
    changes = {}
    for f in feature_names:
        v0 = float(x0[f].iloc[0])
        v1 = float(x_cf[f].iloc[0])

        if abs(v0 - v1) > 1e-6:
            changes[f] = {
                "from": round(v0, 4),
                "to": round(v1, 4),
                "delta": round(v1 - v0, 4),
            }

    cf_pred = float(model.predict_proba(scaler.transform(x_cf))[0][1])

    # ---------- VISUALISATION ----------
    fig = go.Figure()

    if changes:
        fig.add_trace(go.Bar(
            x=list(changes.keys()),
            y=[c["delta"] for c in changes.values()],
            marker_color="#6366f1",
        ))

    fig.update_layout(
        title=f"Counterfactual changes (instance {instance_id})",
        xaxis_title="Feature",
        yaxis_title="Change",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )

    return {
        "data": {
            "instance_id": instance_id,
            "original_prediction": round(original_pred, 4),
            "counterfactual_prediction": round(cf_pred, 4),
            "target": round(target, 4),
            "num_features_changed": len(changes),
            "changes": changes,
        },
        "visualisation": _plotly_payload(
            fig,
            display_mode_bar=False,
            meta={
                "tool": "get_counterfactual_explanation",
                "instance_id": instance_id,
            },
        ),
    }


def get_similar_instances(instance_id: int, k: int = 3):
    """
    Get other instances that are predicted with similar risk 
    """
    _init_if_needed()

    instance_id = int(instance_id)
    k = int(k)

    _check_instance_id(instance_id)

    X_test = _STATE["X_test"]
    X_scaled = _STATE["X_test_scaled"]

    query_vec = X_scaled[instance_id]
    distances = []

    for i in range(len(X_scaled)):
        if i == instance_id:
            continue

        dist = _euclidean_distance(query_vec, X_scaled[i])
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    top = distances[:k]

    rows = []
    indices = []

    for idx, dist in top:
        indices.append(int(idx))
        rows.append({
            "instance_id": int(idx),
            "instance_features": {
                col: float(X_test.iloc[idx][col]) for col in X_test.columns
            },
            "prediction": round(float(_STATE["model"].predict_proba([X_scaled[idx]])[0][1]), 4),
            # "target": float(y_test[idx])
        })

    return {
        "data": {
            "query_instance": instance_id,
            "k": k,
            "indices": indices,
            "instances": rows
        },
        "visualisation": None,
    }


def get_representative_instances(indices: list, k: int = 3):
    """
    Return k representative instances for a filtered subgroup.
    """
    _init_if_needed()
    k = int(k)

    if len(indices) == 0:
        return {"data": "indices must be a non-empty list.", "visualisation": None}

    X_scaled = _STATE["X_test_scaled"]
    X_test = _STATE["X_test"]
    model = _STATE["model"]
    # Ensure k is valid
    k = min(k, len(indices))

    indices = [int(i) for i in indices if 0 <= int(i) < len(X_scaled)]
    if not indices:
        return {"data": "No valid indices provided.", "visualisation": None}

    subgroup_vectors = X_scaled[indices]
    centroid = np.mean(subgroup_vectors, axis=0)
    distances = np.linalg.norm(subgroup_vectors - centroid, axis=1)

    top_positions = np.argsort(distances)[:k]
    rep_indices = [indices[i] for i in top_positions]

    # batch predictions
    preds = model.predict_proba(X_scaled[rep_indices])[:, 1]

    rows = []
    for i, idx in enumerate(rep_indices):
        rows.append({
            "instance_id": int(idx),
            "instance_features": {
                col: float(X_test.iloc[idx][col]) for col in X_test.columns
            },
            "prediction": round(float(preds[i]), 4)
        })

    return {
        "data": {
            "group_size": len(indices),
            "representative_indices": [int(i) for i in rep_indices],
            "instances": rows
        },
        "visualisation": None,
    }

def get_subgroup(filters: dict):
    _init_if_needed()

    X_test = _STATE["X_test"]
    model = _STATE["model"]
    X_scaled = _STATE["X_test_scaled"]

    # If filters accidentally comes as a JSON string, parse it
    filters = _maybe_json_loads(filters)
    if not isinstance(filters, dict):
        return {"data": {"size": 0, "indices": [], "filters": filters, "invalid_filters": ["filters must be a dictionary"]}, "visualisation": None}

    df = X_test.copy()

    # Add predicted risk column so it can be filtered
    preds = model.predict_proba(X_scaled)[:, 1]
    df["prediction"] = preds

    valid_features = set(df.columns)
    invalid_filters = []

    # Validate filters first
    for feat in filters.keys():
        if feat not in valid_features:
            invalid_filters.append(feat)

    # If invalid filter exists -> stop
    if invalid_filters:
        return {
            "data": {"size": 0, "indices": [], "filters": filters, "invalid_filters": invalid_filters,
                "available_features": list(valid_features),},
            "visualisation": None,
        }

    # Use a boolean mask instead of repeatedly filtering dataframe
    mask = np.ones(len(df), dtype=bool)

    for feat, conds in filters.items():
        col = df[feat]

        # Normalise to list
        if not isinstance(conds, list):
            conds = [conds]

        for cond in conds:
            if not isinstance(cond, dict):
                continue

            op = cond.get("op")
            val = cond.get("value")

            if op is None or val is None:
                continue

            # Normalize operator aliases
            op = {
                "gt": ">",
                "ge": ">=",
                "lt": "<",
                "le": "<=",
                "eq": "=="
            }.get(op, op)

            if op == ">":
                mask &= col > val
            elif op == ">=":
                mask &= col >= val
            elif op == "<":
                mask &= col < val
            elif op == "<=":
                mask &= col <= val
            elif op == "==":
                mask &= col == val

    indices = df.index[mask].astype(int).tolist()

    return {
        "data": {
            "size": int(len(indices)),
            "indices": indices,
            "filters": filters,
        },
        "visualisation": None,
    }

def predict_with_feature_changes(instance_id: int, changes: dict):
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    if not isinstance(changes, dict) or not changes:
        return {"data": "changes must be a non-empty object of feature -> new_value.", "visualisation": None}

    model = _STATE["model"]
    scaler = _STATE["scaler"]
    X_test = _STATE["X_test"]

    x0 = X_test.iloc[[instance_id]].copy()
    pred0 = float(model.predict_proba(scaler.transform(x0))[0][1])

    x1 = x0.copy()
    for feat, val in changes.items():
        if feat not in x1.columns:
            return {"data": f"Unknown feature '{feat}'.", "visualisation": None}
        x1[feat] = float(val)

    pred1 = float(model.predict_proba(scaler.transform(x1))[0][1])

    return {
        "data": {
            "instance_id": instance_id,
            "original_prediction": round(pred0, 4),
            "new_prediction": round(pred1, 4),
            "change_in_prediction": float(pred1 - pred0),
            "changes": {k: float(v) for k, v in changes.items()},
            "original_values": {k: float(x0[k].iloc[0]) for k in changes.keys()},
            "new_values": {k: float(x1[k].iloc[0]) for k in changes.keys()},
        },
        "visualisation": None,
    }


# TODO: show the risk prediction distribution
# TODO: only show the feature distribution queried by the users
def dataset_meta(feature: str = None, instance_id: int = None, bins: int = 30):
    """
    Return high-level dataset info.
    If feature is provided, also return that feature's distribution.
    If instance_id is provided, highlight its position.
    """
    _init_if_needed()

    X_train = _STATE["X_train"]
    X_test = _STATE["X_test"]
    y_train = _STATE["y_train"]

    feature_stats = {}
    for col in X_train.columns:
        s = X_train[col].values  
        feature_stats[col] = {
            "mean": round(float(np.mean(s)), 3),
            "min": round(float(np.min(s)), 3),
            "max": round(float(np.max(s)), 3),
            "std": round(float(np.std(s)), 3),
        }

    data = {
        "dataset_name": "credit risk",
        "train_instances": int(len(X_train)),
        "num_features": int(len(X_train.columns)),
        "features": X_train.columns.tolist(),
        "target": "Probability of default",
        "target_statistics": {
            "mean": round(float(np.mean(y_train)), 3),
            "min": round(float(np.min(y_train)), 3),
            "max": round(float(np.max(y_train)), 3),
            "std": round(float(np.std(y_train)), 3),
        },
        "feature_statistics": feature_stats,
    }

    visualisation = None

    # ---------- FEATURE DISTRIBUTION ----------
    if feature is not None:
        if feature not in X_train.columns:
            return {
                "data": {
                    **data,
                    "error": f"Unknown feature '{feature}'."
                },
                "visualisation": None,
            }

        s = X_train[feature].values
        mean_val = float(np.mean(s))
        instance_value = None

        if instance_id is not None:
            instance_id = int(instance_id)
            _check_instance_id(instance_id)
            instance_value = float(X_test.iloc[instance_id][feature])

        # ---------- PLOT ----------
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=s,
            nbinsx=int(bins),
            marker=dict(color="#93c5fd"),
            opacity=0.85,
            hovertemplate=f"{feature}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))

        # Mean line
        fig.add_vline(
            x=mean_val,
            line_width=2,
            line_dash="dot",
            line_color="#1d4ed8",
        )

        fig.add_annotation(
            x=mean_val,
            y=0.8,
            xref="x",
            yref="paper",
            text=f"Mean: {mean_val:.2f}",
            showarrow=False,
        )

        if instance_value is not None:
            fig.add_vline(
                x=instance_value,
                line_width=3,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Your {feature}: {instance_value:.2f}",
                annotation_position="top right",
            )

        fig.update_layout(
            title=f"{feature} distribution",
            xaxis_title=feature,
            yaxis_title="Count",
            bargap=0.05,
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            showlegend=False,
        )

        visualisation = _plotly_payload(
            fig,
            display_mode_bar=False,
            meta={
                "tool": "dataset_meta",
                "feature": feature,
                "instance_id": instance_id,
            },
        )

    return {
        "data": data,
        "visualisation": visualisation,
    }


def model_meta():
    """
    Return high-level information about the trained model.
    """
    _init_if_needed()

    model = _STATE["model"]
    scaler = _STATE["scaler"]

    X_test = _STATE["X_test"]
    y_test = _STATE["y_test"]

    X_scaled = scaler.transform(X_test)

    preds = model.predict_proba(X_scaled)[:, 1]

    accuracy = accuracy_score(y_test, preds > 0.5)

    return {
        # "data": {
            "model_type": "Random Forest Classifier",
            "prediction_task": "Predicting default risk of borrowers",
            "evaluation_metrics": {
                "accuracy": round(accuracy, 4)
            # },
        },
        "visualisation": None,
    }


available_tools_mapping = {
    "generate_local_shap_bar_plot": generate_shap_bar_plot,
    "generate_global_subgroup_shap_plot": generate_shap_summary_plot,
    "get_instance_features_and_prediction": get_instance_features_and_prediction,
    "get_average_prediction": get_average_prediction,
    "get_cp_plot": get_cp_plot,
    "get_counterfactual_explanation": get_counterfactual_explanation,
    "get_subgroup": get_subgroup,
    "predict_with_feature_changes": predict_with_feature_changes,
    "get_similar_instances": get_similar_instances,
    "get_representative_instances": get_representative_instances,
    "dataset_meta": dataset_meta,
    "model_meta": model_meta,
    "get_partial_dependence_plot": get_partial_dependence_plot,
}