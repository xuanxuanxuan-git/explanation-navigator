import json
import threading
import os
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
    "feature_ranges": {
        "Credit used (%)": {"min": 0, "max": 100},
        "Months since last credit application": {"min": 0, "max": 48},
        "On-time payment rate (%)": {"min": 0, "max": 100},
        "Months since last late payment": {"min": 0, "max": 96},
        "Loans not paid off (%)": {"min": 0, "max": 100},
        "Number of loans": {"min": 0, "max": 100},
    }
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "..", "dataset", "heloc_dataset_selected.csv")
DATA_PATH = os.path.normpath(DATA_PATH)

def _init_if_needed(test_size=0.2, random_state=42):
    if _STATE["ready"]:
        return

    with _STATE_LOCK:
        if _STATE["ready"]:
            return
        df = pd.read_csv(DATA_PATH)  
        # Convert special missing codes to NaN first
        df = df.replace([-9, -8, -7], np.nan)

        # Remove rows where non-target feature values are missing
        feature_cols = [c for c in df.columns if c != "CreditScore"]
        df = df.dropna(subset=feature_cols, how="any").reset_index(drop=True)

        # Target: CreditScore (Good/Bad)
        # predicting the risk of being a bad borrower
        y = df["CreditScore"].map({"Good": 0, "Bad": 1}).values
        X = df.drop(columns=["CreditScore"]).copy()

        # Fill remaining missing values with column medians
        # X = X.fillna(X.median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        top_features = ['Credit used (%)', 'Months since last late payment', 'On-time payment rate (%)', 'Number of loans', 'Loans not paid off (%)', 'Months since last credit application']
        
        # Re-scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        background = shap.sample(X_train_scaled, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_scaled[:100], l1_reg=False)[:, :, 0]*100

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
            "ready": True,
        })


def _plotly_payload(fig, *, config=None, meta=None):
    fig_json = json.loads(pio.to_json(fig))
    default_config = {
        "displayModeBar": True,              # Show the bar
        "modeBarButtons": [["toImage"]],     # ONLY show the save image button
        "displaylogo": False                 # Hide the Plotly logo
    }

    if config:
        default_config.update(config)
    return {
        "type": "plotly",
        "figure": fig_json,
        "config": default_config,
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
    colors = ["#ef4444" if v <= 0 else "#3b82f6" for v in values]

    # Calculate range padding to prevent outside text from overlapping the y-axis labels
    min_val = min(values) if values else 0
    max_val = max(values) if values else 0
    
    # Add 25% padding to the min and max values to create space for the outside labels
    padding = (max_val - min_val) * 0.25 if (max_val - min_val) != 0 else 1
    x_range = [min_val - padding, max_val + padding]
    text_labels = [f"{v:+.1f}" for v in values]

    fig = go.Figure(
        data=[go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker={"color": colors},
            customdata=features,
            text=text_labels,
            textfont={"size": 10},
            textposition="outside",
            texttemplate="%{text}",
            cliponaxis=False,
            hovertemplate="Factor: %{y}<br>Contribution: %{x:.1f}<extra></extra>",
        )],
        layout=go.Layout(
            title={
                "text": "What affected your score<br><span style='font-size: 13px; color: gray; font-weight: normal;'>How each factor contributes to your score</span>",
                "y": 0.88,      
                "x": 0.05,      
            },
            xaxis={
                "showticklabels": False,
                "range": x_range,
                "zeroline": False,
            },
            yaxis={
                "ticklabelstandoff": 5, 
            },
            margin={"l": 140, "r": 20, "t": 85, "b": 20}, 
            shapes=[
                dict(
                    type="line",
                    xref="x", x0=0, x1=0,
                    yref="paper", y0=0, y1=1.1, 
                    line=dict(color="black", width=1)
                )
            ],
            annotations=[
                dict(
                    x=0, y=1.15, 
                    xref="x", yref="paper",
                    text="<span style='font-size: 14px;'>←</span> Decrease by",
                    showarrow=False,
                    xanchor="right",
                    xshift=-5,
                    font=dict(size=10, color="gray")
                ),
                dict(
                    x=0, y=1.15, 
                    xref="x", yref="paper",
                    text="Increase by <span style='font-size: 14px;'>→</span>",
                    showarrow=False,
                    xanchor="left",
                    xshift=5,
                    font=dict(size=10, color="gray")
                )
            ]
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
        # TODO: fix -- for now it can only show instances with indices < 100
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

    # Dynamic Main Title and Subtitle based on the source
    main_title = "What mattered most overall" if source == "all" else "What mattered most across selected subgroup"
    subtitle = "How important each factor is across all applicants" if source == "all" else "How important each factor is across these applicants"

    fig = go.Figure(
        data=[go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker={"color": ["#10b981"] * len(x)},
            customdata=y,
            hovertemplate="Factor: %{y}<br>Impact: %{x:.1f}<extra></extra>",
        )],
        layout=go.Layout(
            title={
                "text": f"{main_title}<br><span style='font-size: 13px; color: gray; font-weight: normal;'>{subtitle}</span>",
                "y": 0.88,      
                "x": 0.05, 
            },
            xaxis={"title": {
                    "text": "Average impact on score",
                    "font": {"size": 12} 
                }},
            yaxis={
                "ticklabelstandoff": 5, 
            },
            margin={"l": 120, "r": 20, "t": 75, "b": 20},
        ),
    )
    
    return {
        "data": {
            "source": source,
            "count": int(len(X)),
            "features": rows_sorted,
        },
        "visualisation": _plotly_payload(fig, meta={"tool": "generate_shap_summary_plot"}),
    }


def get_instance_features_and_prediction(instance_id: int):
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    model = _STATE["model"]
    X_test = _STATE["X_test"]
    X_train = _STATE["X_train"]
    X_scaled = _STATE["X_test_scaled"]

    x_raw = X_test.iloc[instance_id]
    x_scaled = X_scaled[[instance_id]]

    pred = float(model.predict_proba(x_scaled)[0][0]) * 100

    feature_info = {}

    for col in X_test.columns:
        train_col = X_train[col].astype(float)

        q1 = float(train_col.quantile(0.01))
        q99 = float(train_col.quantile(0.99))

        val = float(x_raw[col])

        # clip to range
        clipped = max(min(val, q99), q1)

        # normalise 0-1
        if abs(q99 - q1) < 1e-9:
            norm = 0.5
        else:
            norm = (clipped - q1) / (q99 - q1)

        feature_info[col] = {
            "value": val,
            "min": round(q1, 4),
            "max": round(q99, 4),
            "normalised": round(norm, 4),
        }

    return {
        "data": {
            "instance_id": instance_id,
            "features": feature_info,
            "prediction": int(round(pred)),
        },
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
        return {"data": f"Unknown source='{source}'. Use 'all' or 'indices'.", "visualisation": None}

    preds = model.predict_proba(X)[:, 0]*100
    avg = float(np.mean(preds))
    return {
        "data": {
            "source": source,
            "count": int(len(X)),
            "average_credit_score": int(round(avg)),
        },
        "visualisation": None,
    }

def get_cp_plot(instance_id: int, feature: str, grid_points: int = 101):
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    feature = (feature or "").strip()
    grid_points = int(grid_points)

    X_test = _STATE["X_test"]
    scaler = _STATE["scaler"]
    model = _STATE["model"]
    feature_ranges = _STATE["feature_ranges"]

    if feature not in X_test.columns:
        return {"data": f"Unknown feature '{feature}'.", "visualisation": None}

    x0 = X_test.iloc[[instance_id]].copy()
    base_val = float(x0[feature].iloc[0])

    x_min = float(feature_ranges.get(feature).get("min"))
    x_max = float(feature_ranges.get(feature).get("max"))
    grid = np.linspace(x_min, x_max, grid_points)

    preds = []
    for v in grid:
        xv = x0.copy()
        xv[feature] = v
        preds.append(float(model.predict_proba(scaler.transform(xv))[0][0])*100)
    base_pred = float(model.predict_proba(scaler.transform(x0))[0][0]*100)
    fig = go.Figure(
        data=[
            # y=50 line (for consistency with the 'all cp plots' view)
            go.Scatter(
                x=[x_min, x_max],
                y=[50, 50],
                mode="lines",
                line={"color": "#94a3b8", "width": 1.5, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
            ),
            # Line trace for the grid
            go.Scatter(
                x=grid.tolist(),
                y=preds,
                mode="lines",
                line={"color": "#6366f1"},
                hovertemplate=f"{feature}: %{{x:.1f}}<br>Prediction: %{{y:.1f}}<extra></extra>",
                showlegend=False,
            ),
            # Marker trace for the current base value
            go.Scatter(
                x=[base_val],
                y=[base_pred],
                mode="markers",
                marker={"size": 8, "color": "#ef4444"},
                hovertemplate=f"Current {feature}: %{{x:.0f}}<br>Prediction: %{{y:.0f}}<extra></extra>",
                showlegend=False,
            ),
        ],
        layout=go.Layout(
            title={
                "text": f"How changing {feature} <br>affects your score",
                "font": {"size": 16},
                "y": 0.9,
            },
            height=280,
            xaxis={"title": feature, "range": [x_min, x_max], "gridcolor": "white"},
            yaxis={"title": "Credit score", "title_standoff": 5,           
                "range": [0, 100],      
            },   
            margin={"l": 60, "r": 30, "t": 73, "b": 40},
        ),
    )
    # Update y-axis to strictly use specific tickvals
    fig.update_yaxes(
        range=[0, 100], 
        tickmode="array",
        tickvals=[0, 25, 50, 75, 100], 
        gridcolor="white", # Contrasts with blue background
    )
    
    return {
        "data": {
            "instance_id": instance_id,
            "feature": feature,
            "base_value": base_val,
            "sampled_values": [round(v, 1) for v in grid.tolist()],
            "prediction": [round(p, 1) for p in preds],
        },
        "visualisation": _plotly_payload(fig, meta={"tool": "get_cp_plot", "instance_id": instance_id, "feature": feature}),
    }

def generate_all_cp_plots(instance_id: int, grid_points: int = 50):
    """
    Generates Ceteris Paribus (What-If) plots for all features for a given instance.
    Arranges them in a subplot grid.
    """
    _init_if_needed()
    instance_id = int(instance_id)
    _check_instance_id(instance_id)

    grid_points = int(grid_points)

    X_test = _STATE["X_test"]
    scaler = _STATE["scaler"]
    model = _STATE["model"]
    feature_ranges = _STATE["feature_ranges"]

    features = list(X_test.columns)
    num_features = len(features)
    
    # Calculate grid layout (e.g., 2 columns)
    cols = 2
    rows = (num_features + cols - 1) // cols

    # Create subplots
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    x0 = X_test.iloc[[instance_id]].copy()
    base_pred = float(model.predict_proba(scaler.transform(x0))[0][0]*100)

    for i, feature in enumerate(features):
        r = (i // cols) + 1
        c = (i % cols) + 1

        base_val = float(x0[feature].iloc[0])
        x_min = float(feature_ranges.get(feature).get("min"))
        x_max = float(feature_ranges.get(feature).get("max"))
        grid = np.linspace(x_min, x_max, grid_points)

        # Vectorized Predictions (Keeps it lightning fast!)
        xv_batch = x0.loc[x0.index.repeat(grid_points)].copy()
        xv_batch[feature] = grid
        scaled_batch = scaler.transform(xv_batch)
        preds = model.predict_proba(scaled_batch)[:, 0] * 100

        # y=50 line
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[50, 50],
                mode="lines",
                line={"color": "#94a3b8", "width": 1.5, "dash": "dash"},
                hoverinfo="skip",  # Prevents hover tooltip from catching on this line
                showlegend=False,
            ),
            row=r, col=c
        )

        # Add the line trace for the grid
        fig.add_trace(
            go.Scatter(
                x=grid.tolist(),
                y=preds.tolist(),
                mode="lines",
                line={"color": "#6366f1"},
                hovertemplate=f"{feature}: %{{x:.1f}}<br>Prediction: %{{y:.1f}}<extra></extra>",
                showlegend=False,
            ),
            row=r, col=c
        )
        
        # Add the marker trace for the current base value
        fig.add_trace(
            go.Scatter(
                x=[base_val],
                y=[base_pred],
                mode="markers",
                marker={"size": 8, "color": "#ef4444"},
                hovertemplate=f"Current {feature}: %{{x:.0f}}<br>Prediction: %{{y:.0f}}<extra></extra>",
                showlegend=False,
            ),
            row=r, col=c
        )

        # Update y-axis to strictly use specific tickvals
        fig.update_yaxes(
            range=[0, 100], 
            tickmode="array",
            tickvals=[0, 50, 100], 
            gridcolor="white", # Contrasts with blue background
            tickfont={"size": 12},
            row=r, col=c
        )
        
        if c == 1:  # Only add y-axis label on the left-most plots
            fig.update_yaxes(title_text="Credit score", title_standoff=4, row=r, col=c)
            
        font_size = 12 if len(feature) > 15 else 14
        fig.update_xaxes(
            title_text=feature,
            title_font = {"size":font_size},
            range=[x_min, x_max],
            gridcolor="white",
            tickfont={"size": 12},
            row=r, col=c
        )

    # Format the overall layout with subtitle and consistent spacing
    fig.update_layout(
        title={
            "text": "What-If Scenarios<br><span style='font-size: 13px; color: #6b7280; font-weight: normal;'>How changing a single factor changes your score</span>",
            "y": 0.94,
            "x": 0.05,
        },
        height=max(200 * rows, 300), 
        margin={"l": 50, "r": 25, "t": 90, "b": 40}, 
        showlegend=False,
    )

    return {
        "data": {
            "message": "CP plots generated for all features.",
        },
        "visualisation": _plotly_payload(fig, meta={"tool": "generate_all_cp_plots", "instance_id": instance_id}),
    }

def get_partial_dependence_plot(feature: str, grid_points: int = 101):
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
    feature_ranges = _STATE["feature_ranges"]

    if feature not in X_test.columns:
        return {"data": f"Unknown feature '{feature}'.", "visualisation": None}

    feature_idx = X_test.columns.get_loc(feature)

    x_min = float(feature_ranges.get(feature).get("min"))
    x_max = float(feature_ranges.get(feature).get("max"))
    grid = np.linspace(x_min, x_max, grid_points)

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

    # Predict in chunks 
    preds = np.zeros(n * grid_points)
    chunk_size = 10000
    for i in range(0, len(X_rep), chunk_size):
        # Predict on a chunk, extract probability of class 0, and multiply by 100
        preds[i : i + chunk_size] = model.predict_proba(X_rep[i : i + chunk_size])[:, 0] * 100

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
            hovertemplate=f"{feature}: %{{x:.2f}}<br>Avg Prediction: %{{y:.0f}}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": f"Average effect of {feature}<br><span style='font-size: 13px; color: #6b7280; font-weight: normal;'>How this factor affects the average predicted score</span>",
            "font": {"size": 16},
            "y": 0.9,
        },
        xaxis={"title": feature, "range": [x_min, x_max], "gridcolor": "white"},
        yaxis={"title": "Average score", "title_standoff": 5, "range": [0, 100]},
        margin={"l": 60, "r": 30, "t": 73, "b": 40},
    )
    
    # Update y-axis to strictly use specific tickvals
    fig.update_yaxes(
        range=[0, 100], 
        tickmode="array",
        tickvals=[0, 25, 50, 75, 100], 
        gridcolor="white", # Contrasts with blue background
    )
    
    return {
        "data": {
            "feature": feature,
            "sampled_values": [round(v, 1) for v in grid.tolist()],
            "prediction": [int(round(p)) for p in pdp_values.tolist()],
        },
        "visualisation": _plotly_payload(
            fig,
            meta={
                "tool": "get_partial_dependence_plot",
                "feature": feature,
            },
        ),
    }
# TODO: change to a new algorithm
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
    scaler = _STATE["scaler"]
    model = _STATE["model"]
    feature_ranges_state = _STATE["feature_ranges"] 

    x0 = X_test.iloc[[instance_id]].copy()
    x_cf = x0.copy()

    original_pred = float(model.predict_proba(scaler.transform(x0))[0][0]) * 100

    # Default target 
    default_target = False
    if target is None:
        target=50
        default_target = True

    feature_names = X_test.columns.tolist()
    current_pred = original_pred

    # ---------- GREEDY SEARCH ----------
    for _ in range(max_steps):

        best_feature = None
        best_value = None
        best_pred = current_pred

        for f in feature_names:
            min_v = float(feature_ranges_state.get(f).get("min"))
            max_v = float(feature_ranges_state.get(f).get("max"))

            # try small steps in both directions
            candidates = np.linspace(min_v, max_v, 20)

            for v in candidates:
                temp = x_cf.copy()
                temp[f] = v

                pred = float(model.predict_proba(scaler.transform(temp))[0][0])*100

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

    cf_pred = float(model.predict_proba(scaler.transform(x_cf))[0][0])*100

    # ---------- VISUALISATION ----------
    fig = go.Figure()

    features = feature_names  # show ALL features

    def scale(v, f):
        min_v = float(feature_ranges_state.get(f).get("min"))
        max_v = float(feature_ranges_state.get(f).get("max"))
        
        if max_v - min_v < 1e-9:
            return 0.5  # constant feature safeguard
        # clip to avoid going outside range
        v = max(min(v, max_v), min_v)
        return (v - min_v) / (max_v - min_v)

    for i, f in enumerate(features):
        v0_raw = float(x0[f].iloc[0])
        v1_raw = float(x_cf[f].iloc[0])

        v0 = scale(v0_raw, f)
        v1 = scale(v1_raw, f)

        changed = abs(v0_raw - v1_raw) > 1e-6

        # --- vertical band (range) ---
        fig.add_trace(go.Scatter(
            x=[f, f],
            y=[v0, v1],
            mode="lines",
            line=dict(
                color="rgba(99,102,241,0.4)" if changed else "rgba(180,180,180,0.3)",
                width=10 if changed else 6,
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # --- arrow showing direction ---
        if changed:
            fig.add_annotation(
                x=f,
                y=v1,
                ax=f,
                ay=v0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=1,
                arrowsize=1.5,
                arrowwidth=1.5,
                arrowcolor="rgba(99,102,241,0.9)",
                opacity=0.9
            )

            # --- counterfactual point ---
            fig.add_trace(go.Scatter(
                x=[f],
                y=[v1],
                mode="text",
                text=[f"{v1_raw:.0f}"],
                textposition="top center" if v1>=v0 else "bottom center",
                textfont=dict(
                    size=10,
                    color="black"
                ),
                hovertemplate=(
                    f"<b>{f}</b><br>"
                    f"Counterfactual: {v1_raw:.0f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))

        # --- original point (the black horizontal line) ---
        fig.add_trace(go.Scatter(
            x=[f],
            y=[v0],
            mode="markers+text",
            marker=dict(
                symbol="line-ew", 
                color="black",
                size=10,            # increase size so the line is visible
                opacity=1 if changed else 0.6,
                line=dict(width=1, color="black")  
            ),
            text=[f"{v0_raw:.0f}"],
            textposition="bottom center" if v1>=v0 else "top center",
            textfont=dict(
                size=10,
                color="black"
            ),
            hovertemplate=(
                f"<b>{f}</b><br>"
                f"Original: {v0_raw:.0f}<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # Dynamic main title reflecting the target score
    target_display = int(round(target))
    
    fig.update_layout(
        title={
            "text": f"How to improve your score to {target_display}<br><span style='font-size: 13px; color: gray; font-weight: normal;'>The smallest changes needed to reach the target score</span>",
            "y": 0.93,      
            "x": 0.05, 
        },
        # Increased 't' (top margin) to accommodate subtitle
        margin={"l": 40, "r": 20, "t": 65, "b": 120},
        template="plotly_white",
        showlegend=False,
    )

    fig.update_yaxes(
        visible=False,
        range=[-0.2, 1.15]  # allow space for text below
    )

    return {
        "data": {
            "instance_id": instance_id,
            "original_prediction": int(round(original_pred)),
            "counterfactual_prediction": int(round(cf_pred)),
            "target": target,
            "num_features_changed": len(changes),
            "changes": changes,
            "system_reminder": "The user did not specify a target score. You MUST explicitly inform the user that a default target of 50 was assumed." if default_target else ""
        },
        "visualisation": _plotly_payload(
            fig,
            meta={
                "tool": "get_counterfactual_explanation",
                "instance_id": instance_id,
                "target": target,
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
            "prediction": int(round(float(_STATE["model"].predict_proba([X_scaled[idx]])[0][0])*100)),
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
    preds = model.predict_proba(X_scaled[rep_indices])[:, 0]*100

    rows = []
    for i, idx in enumerate(rep_indices):
        rows.append({
            "instance_id": int(idx),
            "instance_features": {
                col: float(X_test.iloc[idx][col]) for col in X_test.columns
            },
            "prediction": int(round(float(preds[i])))
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
    preds = model.predict_proba(X_scaled)[:, 0]*100
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
    pred0 = float(model.predict_proba(scaler.transform(x0))[0][0])*100

    x1 = x0.copy()
    for feat, val in changes.items():
        if feat not in x1.columns:
            return {"data": f"Unknown feature '{feat}'.", "visualisation": None}
        x1[feat] = float(val)

    pred1 = float(model.predict_proba(scaler.transform(x1))[0][0])*100

    return {
        "data": {
            "instance_id": instance_id,
            "original_prediction": int(round(pred0)),
            "new_prediction": int(round(pred1)),
            "change_in_prediction": int(round(pred1 - pred0)),
            "changes": {k: float(v) for k, v in changes.items()},
            "original_values": {k: float(x0[k].iloc[0]) for k in changes.keys()},
            "new_values": {k: float(x1[k].iloc[0]) for k in changes.keys()},
        },
        "visualisation": None,
    }


# TODO: show the risk prediction distribution
# TODO: only show the feature distribution queried by the users
# TODO: show the feature distribution of a specific user group
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
            "mean": round(float(np.mean(s)), 0),
            "min": round(float(np.min(s)), 0),
            "max": round(float(np.max(s)), 0),
            "std": round(float(np.std(s)), 0),
        }

    data = {
        "dataset_name": "credit score",
        "train_instances": int(len(X_train)),
        "num_features": int(len(X_train.columns)),
        "features": X_train.columns.tolist(),
        "target": "Credit score",
        "target_statistics": {
            "mean": round(float(np.mean(y_train)), 0),
            "min": round(float(np.min(y_train)), 0),
            "max": round(float(np.max(y_train)), 0),
            "std": round(float(np.std(y_train)), 0),
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
            hovertemplate=f"{feature}: %{{x:.0f}}<br>Count: %{{y}}<extra></extra>",
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
            text=f"Mean: {mean_val:.0f}",
            showarrow=False,
            xanchor="left",
            xshift=5,
        )

        if instance_value is not None:
            fig.add_vline(
                x=instance_value,
                line_width=3,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Your {feature}: {instance_value:.0f}",
                annotation_position="top right",
            )

        fig.update_layout(
            title={
                "text": f"{feature} distribution<br><span style='font-size: 13px; color: gray; font-weight: normal;'>How this factor is distributed across all applicants</span>",
                "font": {"size": 16},
                "y": 0.9,
                "yanchor": "top",
                "x": 0.05,
            },
            xaxis={"title": feature, "gridcolor": "white"},
            yaxis={"title": "No of applicants", "gridcolor": "white"},
            bargap=0.05,
            # Increased top margin to fit the subtitle
            margin={"l": 60, "r": 20, "t": 70, "b": 20},
            showlegend=False,
        )

        visualisation = _plotly_payload(
            fig,
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

    preds = model.predict_proba(X_scaled)[:, 0] * 100

    accuracy = accuracy_score(y_test, preds > 0.5)

    return {
        # "data": {
            "model_type": "Neural Network Classifier",
            "prediction_task": "Predicting credit score of applicants",
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
    "generate_all_cp_plots": generate_all_cp_plots,
    "get_counterfactual_explanation": get_counterfactual_explanation,
    "get_subgroup": get_subgroup,
    "predict_with_feature_changes": predict_with_feature_changes,
    "get_similar_instances": get_similar_instances,
    "get_representative_instances": get_representative_instances,
    "dataset_meta": dataset_meta,
    "model_meta": model_meta,
    "get_partial_dependence_plot": get_partial_dependence_plot,
}