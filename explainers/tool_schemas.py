# Define tool schemas
from typing import Dict, Union
from pydantic import BaseModel, Field

# define the tool properties with pydantic
class ShapBarPlot(BaseModel):
    # Make it required (no default) so JSON schema includes it in "required" reliably. [web:296]
    instance_id: int = Field(description="Index of the instance to generate SHAP bar plot for (0-based).")
    # max_display: int = Field(default=10, description="Max number of features to show.")

# class ShapSummaryPlot(BaseModel):
    # max_display: int = Field(default=10, description="Max number of features to show in the global summary.")

class IndividualPrediction(BaseModel):
    instance_id: int = Field(description="Index of the instance to predict for (0-based).")

class CpPlot(BaseModel):
    instance_id: int = Field(description="Index of the instance (0-based).")
    feature: str = Field(description="Feature name to vary for the CP curve.")
    # grid_points: int = Field(default=30, description="Number of grid points to evaluate.")

class Condition(BaseModel):
    op: str = Field(
        description="Comparison operator. One of: >, >=, <, <=, ==, gt, ge, lt, le, eq"
    )
    value: float = Field(description="Numeric value to compare against.")

class Subgroup(BaseModel):
    filters: Dict[str, Condition] = Field(
        description="Map of feature name to filtering condition."
    )

class PredictWithFeatureChanges(BaseModel):
    instance_id: int = Field(description="Index of the instance to edit (0-based).")
    changes: Dict[str, float] = Field(
        description=(
            "Map of feature name to new numeric value. Examples: {'AveRooms': 2.0} if change number of rooms to 2. {'HouseAge', 3.0} if change house age to 3."
        )
    )

explainer_tools = [
    {
        "type": "function",
        "name": "generate_local_shap_bar_plot",
        "description": "Provides LOCAL feature importance for a SINGLE instance using SHAP values. "
        "Use this when the user asks why the model made a prediction for a specific instance, "
        "row, example, or ID (e.g., 'instance 1', 'this prediction'). "
        "The plot shows how each feature contributed to that instance's prediction. "
        "This explanation applies only to the selected instance and does NOT represent "
        "feature importance across the dataset.", 
        "parameters": ShapBarPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "generate_global_shap_summary_plot",
        "description": "Provides GLOBAL feature importance across the entire dataset using SHAP values. "
        "Use this when the user asks which features are most important overall in the model "
        "or across all instances (e.g., 'Which features matter most?', 'What are the most "
        "important features in the model?'). The plot aggregates SHAP values across the "
        "dataset to show which features generally have the largest impact on predictions. "
        "This explanation reflects overall model behaviour and should NOT be used for "
        "questions about a specific instance, row, or prediction.", 
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_individual_prediction",
        "description": "Return the model prediction for a specific instance.",
        "parameters": IndividualPrediction.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_average_prediction",
        "description": "Compute average prediction over all test instances or over a provided list of indices (for example, indices returned by get_subgroup).",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["all", "indices"],
                    "description": "Use 'all' to average over the full test set, or 'indices' to average over the provided indices.",
                    "default": "all",
                },
                "indices": {
                    "type": "array",
                    "description": "List of instance indices to include when source='indices'.",
                    "items": {"type": "integer"},
                },
            },
            "required": ["source"],
        },
    },
    {
        "type": "function",
        "name": "get_cp_plot",
        "description": "Generate a CP plot for one instance over one feature (vary feature across a grid; other features fixed).",
        "parameters": CpPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_subgroup",
        "description": """
            Filter test instances by feature conditions and return indices.
            Examples:
            - "houses with less than 1 bedroom"
            -> {"filters": {"AveBedrms": {"op": "<", "value": 1}}}

            - "MedInc greater than 5"
            -> {"filters": {"MedInc": {"op": ">", "value": 5}}}

            - "AveRooms <= 3"
            -> {"filters": {"AveRooms": {"op": "<=", "value": 3}}}
            """,
        "parameters": Subgroup.model_json_schema(),
    },
    {
        "type": "function",
        "name": "predict_with_feature_changes",
        "description": """
            Alter one or more feature values for an instance and return the new prediction
            MUST provide both instance_id AND changes.
            Example for 'change bedrooms to 1 for instance 2': {\"instance_id\": 2, \"changes\": {\"AveBedrms\": 1}}
        """,
        "parameters": PredictWithFeatureChanges.model_json_schema(), 
    },
    {
        "type": "function",
        "name": "get_counterfactual_explanation",
        "description": "Retrieve counterfactual explanations.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

