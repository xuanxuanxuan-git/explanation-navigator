# Define tool schemas
from typing import Dict, Union, List
from pydantic import BaseModel, Field

# define the tool properties with pydantic
class ShapBarPlot(BaseModel):
    # Make it required (no default) so JSON schema includes it in "required" reliably. [web:296]
    instance_id: int = Field(description="Index of the instance to generate SHAP bar plot for (0-based).")
    # max_display: int = Field(default=10, description="Max number of features to show.")

class ShapSummaryPlot(BaseModel):
    source: str = Field(default="all",
        description="Use 'all' to compute feature importance for the entire dataset, or 'indices' to compute importance for a subgroup."
    )
    indices: list[int] | None = Field(default=None,
        description="Optional list of instance indices when source='indices'. Usually obtained from get_subgroup."
    )
    max_display: int = Field(
        default=10,
        description="Maximum number of features to show in the SHAP summary plot."
    )

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

class SimilarInstances(BaseModel):
    instance_id: int = Field(
        description="Index of the query instance (0-based) whose similar instances should be retrieved."
    )
    k: int = Field(
        default=3,
        description="Number of similar instances to return."
    )

class RepresentativeInstances(BaseModel):
    indices: List[int] = Field(
        description=(
            "List of dataset indices representing a subgroup of instances. These indices are usually obtained from the get_subgroup tool."
        )
    )
    k: int = Field(default=3, description="Number of representative instances to return from the subgroup.")

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
        "name": "generate_global_subgroup_shap_plot",
        "description": """
            Provides importance of features on the model's predictions across the entire dataset or a subset using SHAP values. 

            It can compute importance for the entire dataset (global importance), or a subgroup of instances (for example houses with certain characteristics).

            Use this when the user asks questions such as:
            - "Which features are most important in the model?"
            - "What features matter most overall?"
            - "For houses with more than 5 rooms, which features influence the price the most?"
            
            Do NOT used this tool for questions about a specific instance.", 
        """,
        "parameters": ShapSummaryPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_individual_prediction",
        "description": "Return the predicted house price for a specific instance, along with all the feature values of that instance.",
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
            Filter test instances by feature conditions and return the indices of instances that satisfy the conditions.

            Each filter specifies a feature name and a comparison condition. The feature name must match a dataset column
            or the special field "predicted_price", which refers to the model's predicted value.

            Supported comparison operators: >, >=, <, <=, ==

            Filter format: {"filters": {"<feature_name>": {"op": "<operator>", "value": <number>}}}

            Multiple filters can be combined; all conditions must be satisfied (logical AND).

            Examples:
            User query: "houses with less than 1 bedroom"
            -> {"filters": {"AveBedrms": {"op": "<", "value": 1}}}

            User query: "houses where median income is greater than 5"
            -> {"filters": {"MedInc": {"op": ">", "value": 5}}}

            User query: "houses with predicted price above 1.3"
            -> {"filters": {"predicted_price": {"op": ">", "value": 1.3}}}

            User query: "houses with more than 5 rooms and price below 2"
            -> {
                "filters": {
                    "AveRooms": {"op": ">", "value": 5},
                    "predicted_price": {"op": "<", "value": 2}
                }
            }

            If the user asks about a subgroup of houses or wants to compute statistics for houses satisfying certain conditions,
            this tool should be used first to retrieve the matching instances.
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
        "name": "get_similar_instances",
        "description": (
            "Find instances in the dataset that are most similar to a given instance based on their feature values. "
            "Use this when the user asks for examples similar to a specific house or wants to compare an instance with other similar instances. "
            "The tool returns the indices and features of the most similar instances."
        ),
        "parameters": SimilarInstances.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_representative_instances",
        "description": (
            "Return representative examples from a subgroup of the dataset. "
            "The subgroup must first be identified using the get_subgroup tool, "
            "which returns a list of indices. These indices should then be passed "
            "to this function. The function selects the k instances closest to the "
            "subgroup centroid in feature space, representing typical examples of that group."
        ),
        "parameters": RepresentativeInstances.model_json_schema(),
    },
    {
        "type": "function",
        "name": "dataset_meta",
        "description": (
            "Provide general information about the dataset used by the AI model. "
            "Use this when the user asks questions like "
            "'What data was used to train the model?', "
            "'How many houses are in the dataset?', or "
            "'What features describe the houses?'. "
            "The tool returns dataset size, feature names, and statistics of each feature such as average, min, max."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
    },
    {
        "type": "function",
        "name": "model_meta",
        "description": (
            "Provide general information about the AI model used to make predictions. "
            "Use this when the user asks about how the model works, what algorithm it uses, or how accurate it is. "
            "The tool returns the model type, prediction task, and evaluation metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
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

