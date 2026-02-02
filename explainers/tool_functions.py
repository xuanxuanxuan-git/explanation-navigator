
# def generate_shap_bar():
#     pass

# def generate_shap_summary():
#     pass

# def get_feature_attribution_ranking(instance_id: int):
#     """
#     Returns a sorted ranking (descending) of features and their SHAP attribution values for a specific instance.
#     """

#     instance_id = int(instance_id)

#     # # extract SHAP values for this instance
#     # shap_vals = shap_values[instance_id]  # array-like of shape (n_features,)
#     # feature_names = X_test.columns.tolist()

#     # # build a list of (feature_name, shap_value) pairs
#     # ranking = list(zip(feature_names, shap_vals))

#     # # sort by descending absolute SHAP importance
#     # ranking_sorted = sorted(ranking, key=lambda x: abs(x[1]), reverse=True)
    
#     # result = {"description": [
#     #     {"feature": feature, "shap_value": float(value)}
#     #     for feature, value in ranking_sorted
#     # ]}

#     data = [
#         {"feature": "MedInc", "shap_value": -0.4597899429327556},
#         {"feature": "Latitude", "shap_value": -0.3695846485876284},
#         {"feature": "AveOccup", "shap_value": -0.08532596763202996},
#         {"feature": "AveRooms", "shap_value": -0.07967135426168119},
#         {"feature": "Longitude", "shap_value": 0.04012051924944584},
#     ]

#     x = [d["feature"] for d in data]
#     y = [d["shap_value"] for d in data]

#     fig = go.Figure(
#         data=[
#             go.Bar(
#                 x=x,
#                 y=y,
#                 name="SHAP values",
#                 marker={"color": ["#3b82f6"] * len(x)},
#                 customdata=x,  # helps identify bars client-side
#                 hovertemplate="Feature: %{x}<br>SHAP: %{y}<extra></extra>",
#             )
#         ],
#         layout=go.Layout(
#             title=f"Feature attribution for instance id={instance_id}",
#             xaxis={"title": "Feature"},
#             yaxis={"title": "SHAP value"},
#             margin={"l": 40, "r": 20, "t": 50, "b": 40},
#         ),
#     )
    
#     # Convert Plotly figure to JSON-serializable dict.
#     fig_json_str = pio.to_json(fig)
#     fig_json = json.loads(fig_json_str)
    
#     return {
#         "data": data,                  
#         "visualisation": {             
#             "type": "plotly",
#             "figure": fig_json,        # contains {data, layout, ...} per Plotly schema
#             "config": {"displayModeBar": False, "responsive": True},
#         },
#     }


# def predict_outcome():
#     probability = 0.54
#     return probability

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
    "X_test": None,          # pd.DataFrame
    "shap_values": None,     # np.ndarray (n_test, n_features) for regression
}

def _init_if_needed(n_points=1000, test_size=0.2, random_state=42):
    """
    One-time initialization: load data, train model, compute SHAP values.
    Keeps it cached for fast tool calls.
    """
    if _STATE["ready"]:
        return

    with _STATE_LOCK:
        if _STATE["ready"]:
            return

        X, y = shap.datasets.california(n_points=n_points)  # returns DataFrame
        X = pd.DataFrame(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train_scaled, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        # Cache
        _STATE["X_test"] = X_test.reset_index(drop=True)
        _STATE["shap_values"] = np.array(shap_values)
        _STATE["ready"] = True


def _plotly_figure_payload(fig, *, display_mode_bar=False):
    fig_json = json.loads(pio.to_json(fig))
    return {
        "type": "plotly",
        "figure": fig_json,
        "config": {"displayModeBar": display_mode_bar, "responsive": True},
    }


# ----------------------------
# Tools (match your mapping)
# ----------------------------

def generate_shap_bar_plot(instance_id: int):
    """
    Bar chart for local attribution (top features by absolute SHAP).
    Returns Plotly figure + underlying data for LLM.
    """
    _init_if_needed()

    instance_id = int(instance_id)
    X_test = _STATE["X_test"]
    shap_values = _STATE["shap_values"]

    if instance_id < 0 or instance_id >= len(X_test):
        return {
            "data": f"Invalid instance_id={instance_id}. Must be 0..{len(X_test)-1}.",
            "visualisation": None
        }

    feature_names = X_test.columns.tolist()
    shap_vals = shap_values[instance_id].tolist()

    rows = [{"feature": f, "shap_value": float(v)} for f, v in zip(feature_names, shap_vals)]
    rows_sorted = sorted(rows, key=lambda r: abs(r["shap_value"]), reverse=True)[:10]

    x = [r["feature"] for r in rows_sorted]
    y = [r["shap_value"] for r in rows_sorted]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                name="Local SHAP",
                marker={"color": ["#3b82f6"] * len(x)},
                customdata=x,
                hovertemplate="Feature: %{x}<br>SHAP: %{y:.4f}<extra></extra>",
            )
        ],
        layout=go.Layout(
            title=f"Local SHAP bar plot (instance_id={instance_id})",
            xaxis={"title": "Feature"},
            yaxis={"title": "SHAP value"},
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        ),
    )

    return {
        "data": rows_sorted,
        "visualisation": _plotly_figure_payload(fig, display_mode_bar=False),
    }


def generate_shap_summary_plot():
    """
    Global importance: mean(|SHAP|) over all instances.
    (This avoids matplotlib summary_plot and keeps it Plotly-friendly.)
    """
    _init_if_needed()

    X_test = _STATE["X_test"]
    shap_values = _STATE["shap_values"]

    feature_names = X_test.columns.tolist()
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    rows = [{"feature": f, "mean_abs_shap": float(v)} for f, v in zip(feature_names, mean_abs)]
    rows_sorted = sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)[:15]

    x = [r["feature"] for r in rows_sorted][::-1]
    y = [r["mean_abs_shap"] for r in rows_sorted][::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=y,
                y=x,
                orientation="h",
                name="Mean |SHAP|",
                marker={"color": ["#10b981"] * len(x)},
                customdata=x,
                hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            )
        ],
        layout=go.Layout(
            title="Global feature importance (mean |SHAP|)",
            xaxis={"title": "Mean |SHAP|"},
            yaxis={"title": "Feature"},
            margin={"l": 90, "r": 20, "t": 50, "b": 40},
        ),
    )

    return {
        "data": rows_sorted,
        "visualisation": _plotly_figure_payload(fig, display_mode_bar=False),
    }


def get_feature_attribution_ranking(instance_id: int):
    """
    Ranking (descending by abs SHAP) for a specific instance.
    Returns both table data and a Plotly bar chart (optional but useful).
    """
    _init_if_needed()

    instance_id = int(instance_id)
    X_test = _STATE["X_test"]
    shap_values = _STATE["shap_values"]

    if instance_id < 0 or instance_id >= len(X_test):
        return {
            "data": f"Invalid instance_id={instance_id}. Must be 0..{len(X_test)-1}.",
            "visualisation": None
        }

    feature_names = X_test.columns.tolist()
    shap_vals = shap_values[instance_id].tolist()

    rows = [{"feature": f, "shap_value": float(v)} for f, v in zip(feature_names, shap_vals)]
    rows_sorted = sorted(rows, key=lambda r: abs(r["shap_value"]), reverse=True)

    return {
        "data": rows_sorted,
        "visualisation": None, 
    }


def predict_price_probability():
    # keep your placeholder; later you can compute from model predictions
    return {
        "data": {"probability_over_30k": 0.54},
        "visualisation": None,
    }


available_tools_mapping = {
        "generate_shap_bar_plot": generate_shap_bar_plot,
        "generate_shap_summary_plot": generate_shap_summary_plot,
        "get_feature_attribution_ranking": get_feature_attribution_ranking, 
        "predict_price_probability": predict_price_probability,
    }
        