import plotly.graph_objects as go
import plotly.io as pio
import json


def generate_shap_bar():
    pass

def generate_shap_summary():
    pass


def get_feature_attribution_ranking(instance_id, name):
    """
    Returns a sorted ranking (descending) of features and their SHAP attribution values for a specific instance.
    """

    instance_id = int(instance_id)

    # # extract SHAP values for this instance
    # shap_vals = shap_values[instance_id]  # array-like of shape (n_features,)
    # feature_names = X_test.columns.tolist()

    # # build a list of (feature_name, shap_value) pairs
    # ranking = list(zip(feature_names, shap_vals))

    # # sort by descending absolute SHAP importance
    # ranking_sorted = sorted(ranking, key=lambda x: abs(x[1]), reverse=True)
    
    # result = {"description": [
    #     {"feature": feature, "shap_value": float(value)}
    #     for feature, value in ranking_sorted
    # ]}

    data = [
        {"feature": "MedInc", "shap_value": -0.4597899429327556},
        {"feature": "Latitude", "shap_value": -0.3695846485876284},
        {"feature": "AveOccup", "shap_value": -0.08532596763202996},
        {"feature": "AveRooms", "shap_value": -0.07967135426168119},
        {"feature": "Longitude", "shap_value": 0.04012051924944584},
    ]

    x = [d["feature"] for d in data]
    y = [d["shap_value"] for d in data]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                name="SHAP values",
                marker={"color": ["#3b82f6"] * len(x)},
                customdata=x,  # helps identify bars client-side
                hovertemplate="Feature: %{x}<br>SHAP: %{y}<extra></extra>",
            )
        ],
        layout=go.Layout(
            title=f"Feature attribution for {name} (id={instance_id})",
            xaxis={"title": "Feature"},
            yaxis={"title": "SHAP value"},
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        ),
    )

    # Convert Plotly figure to JSON-serializable dict.
    fig_json_str = pio.to_json(fig)
    fig_json = json.loads(fig_json_str)

    return {
        "data": data,                  
        "visualisation": {             
            "type": "plotly",
            "figure": fig_json,        # contains {data, layout, ...} per Plotly schema
            "config": {"displayModeBar": False, "responsive": True},
        },
    }


def predict_outcome():
    probability = 0.54
    return probability

available_tools_mapping = {
        "generate_shap_bar_plot": generate_shap_bar,
        "generate_shap_summary_plot": generate_shap_summary,
        "get_feature_attribution_ranking": get_feature_attribution_ranking, 
        "predict_price_probability": predict_outcome,
    }
        