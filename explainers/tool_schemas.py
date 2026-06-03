# Define tool schemas
from typing import Dict, Union, List
from pydantic import BaseModel, Field
from typing import Optional

# define the tool properties with pydantic
class ShapBarPlot(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based). Returns local SHAP explanation for this individual's predicted credit score.")

class ShapSummaryPlot(BaseModel):
    source: str = Field(
        default="all",
        description="Use 'all' for global feature importance across all applicants, or 'indices' for a subgroup."
    )
    indices: Optional[List[int]] = Field(
        default=None,
        description="List of applicant indices when source='indices'. Usually obtained from get_subgroup."
    )

class IndividualPrediction(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based). Returns their features and predicted credit score.")

class AveragePrediction(BaseModel):
    source: str = Field(
        default="all",
        description="Use 'all' to average over the full test set, or 'indices' to average over a selected subgroup."
    )
    indices: Optional[List[int]] = Field(
        default=None,
        description="List of applicant indices to include when source='indices'."
    )

class CpPlot(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based).")
    feature: str = Field(description="Feature to vary (e.g. Credit used (%), Months since last late payment, On-time payment rate (%)).")

class Condition(BaseModel):
    op: str = Field(
        description="Comparison operator. One of: >, >=, <, <=, ==, gt, ge, lt, le, eq"
    )
    value: float = Field(description="Numeric value to compare against.")

class Subgroup(BaseModel):
    filters: Dict[str, Union[Condition, List[Condition]]] = Field(
        description=(
            "Map of feature name to filtering condition(s). "
            "Each feature can have a single condition or a list of conditions. "
            "If multiple conditions are provided, all must be satisfied (logical AND)."
        )
    )

class PredictWithFeatureChanges(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based).")
    changes: Dict[str, float] = Field(
        description=(
            "Map of feature name to new numeric value. Examples: {'Number of loans': 80} if changing Number of loans to 80."
        )
    )

class CounterfactualExplanation(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based).")
    target: Optional[float] = Field(
        description="Desired credit score (0-100, >50 means approved)."
    )

class SimilarInstances(BaseModel):
    instance_id: int = Field(
        description="Index of the query applicant (0-based) whose similar instances should be retrieved."
    )
    k: int = Field(
        default=3,
        description="Number of similar applicants to return."
    )

class RepresentativeInstances(BaseModel):
    indices: List[int] = Field(
        description=(
            "List of dataset indices representing a subgroup of applicants. These indices are usually obtained from the get_subgroup tool."
        )
    )
    k: Optional[int] = Field(default=3, description="Number of representative instances to return from the subgroup.")

class DatasetMeta(BaseModel):
    feature: Optional[str] = Field(
        default=None,
        description=(
            "Feature name or prediction to analyse. Must match an existing feature exactly or 'prediction'."
        )
    )
    instance_id: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Index of the applicant (0-based)."
        )
    )

class PartialDependencePlot(BaseModel):
    feature: str = Field(description="Feature name for which to compute the partial dependence plot.")


explainer_tools = [
    {
        "type": "function",
        "name": "generate_local_shap_bar_plot",
        "description": "Provides LOCAL feature importance for a SINGLE applicant using SHAP values. "
        "Use this when the user asks why the model predicted a high or low credit score for a specific applicant. "
        "The plot shows how each feature contributed to that instance's prediction. "
        "This explanation applies only to the selected applicant and does NOT represent "
        "feature importance across the dataset.", 
        "parameters": ShapBarPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "generate_global_subgroup_shap_plot",
        "description": """
            Provides importance of features on the model's predictions across the entire dataset or a subset using SHAP values. 
            It can compute importance for the entire dataset (global importance), or a subgroup of instances (for example applicants with certain characteristics).
            Use this when the user asks questions such as:
            - "Which features are most important in the model?"
            - "What features matter most overall?"
            - "For applicants with more than 12 months since last late payment, which features influence their predicted credit score the most?"
            
            Do NOT used this tool for questions about a specific instance.", 
        """,
        "parameters": ShapSummaryPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_instance_features_and_prediction",
        "description": (
            "Returns the feature values and predicted credit score for a SINGLE applicant. "
            
            "Use this when the user asks about one applicant's data, such as: "
            "- 'what are my feature values?' "
            "- 'what is my predicted credit score?' "
            "- 'why is my score high/low?' or use it when other tools need to access an applicant's feature values. "
 
            "This tool is often used together with local explanation tools (e.g., generate_local_shap_bar_plot) to provide complete explanations."
        ),  
        "parameters": IndividualPrediction.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_average_prediction",
        "description": "Compute average predicted credit score over all test applicants or over a provided list of applicant indices (for example, indices returned by get_subgroup).",
        "parameters": AveragePrediction.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_cp_plot",
        "description": "Generate a Ceteris Paribus (CP) plot for one applicant and one feature. This shows how the applicant's predicted credit score changes when one feature varies "
        "while all other features are kept fixed at their original values. "
        "Use this tool when the user asks:"
            "- 'How does this feature affect my outcome?'"
            "- 'What happens if I change my Number of loans'"
            "- 'What if this feature was higher or lower?'"
        "Note: This is a local explanation for ONE instance, and only ONE feature is varied at a time. Feature must match a valid dataset feature name exactly.",
        "parameters": CpPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_subgroup",
        "description": """
            Filter test applicants by feature conditions and return the indices of applicants who satisfy the conditions.

            Each filter specifies a feature name and a comparison condition. The feature name must match a dataset column. You may also filter on 'prediction'. 

            Supported comparison operators: >, >=, <, <=, ==

            Filter format:
            {
                "filters": {
                    "<feature_name>": {"op": "<operator>", "value": <number>}
                }
            }
            OR for multiple conditions on the same feature:
            {
                "filters": {
                    "<feature_name>": [
                        {"op": ">=", "value": 67},
                        {"op": "<=", "value": 71}
                    ]
                }
            }
            Multiple filters across features are combined with logical AND.

            Examples:
            "applicants with recent late payments"
            -> {"filters": {"Months since last late payment": {"op": "<", "value": 12}}}

            "applicants with on-time payment rate between 80% and 90%"
            -> {"filters": {"On-time payment rate (%)": [
                    {"op": ">=", "value": 80},
                    {"op": "<=", "value": 90}
            ]}}

            "excellent payment history but low predicted score"
            -> {"filters": {
                    "On-time payment rate (%)": {"op": ">", "value": 95},
                    "prediction": {"op": "<", "value": 50}
                }
            }

            If the user asks about a subgroup of applicants or wants to compute statistics for applicants satisfying certain conditions, this tool should be used first to retrieve the matching applicants.
            """,
        "parameters": Subgroup.model_json_schema(),
    },
    {
        "type": "function",
        "name": "predict_with_feature_changes",
        "description": """
            Alter one or more feature values for a single applicant and return the new predicted credit score. Use this for what-if analysis such as changing an applicant's financial attributes. 
            Example for 'change on-time payment rate to 95 for instance 2': {'instance_id': 2, 'changes': {'On-time payment rate (%)': 95}}
        """,
        "parameters": PredictWithFeatureChanges.model_json_schema(), 
    },
    {
        "type": "function",
        "name": "get_similar_instances",
        "description": (
            "Find applicants in the dataset that are most similar to a given applicant based on their feature values. "
            "Use this when the user wants comparable applicants or similar credit profiles. "
            "The tool returns the indices and profiles (feature values) of the most similar applicants."
        ),
        "parameters": SimilarInstances.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_representative_instances",
        "description": (
            "Return representative applicants from a subgroup of the dataset. "
            "The subgroup must first be identified using the get_subgroup tool, "
            "which returns a list of indices. These indices should then be passed "
            "to this function. The function selects the k applicants closest to the "
            "subgroup centroid in feature space, representing typical examples of that group."
        ),
        "parameters": RepresentativeInstances.model_json_schema(),
    },
    {
        "type": "function",
        "name": "dataset_meta",
        "description": (
            "Provide information about the credit scoring dataset used by the AI model. "
            "Use this when the user asks about the dataset overview (size, features, target), "
            "summary statistics, or the distribution of a specific feature. "
            "If 'feature' is provided, return that feature's distribution statistics and a distribution chart. "
            "If both 'feature' and 'instance_id' are provided, also show where that applicant's feature value lies in the distribution."
        ),
        "parameters": DatasetMeta.model_json_schema(),
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
        "description": (
            "Generate a counterfactual explanation for a given applicant. A counterfactual explanation provides a set of feature changes that must be applied TOGETHER to move the predicted credit score toward a target value. "
            "Note that the returned changes must be implemented simultaneously to reach the target prediction. Individual changes should NOT be interpreted in isolation. "
            "Only include `target` if the user explicitly specifies a desired score."
            "Use this tool when the user asks: "
            "- 'What should I change to increase my score?' -> {'instance_id': }"
            "- 'How can I increase the predicted score to 60?' -> {'instance_id': , 'target': 60}"
            "- 'What would make my score higher?'"
        ),
        "parameters": CounterfactualExplanation.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_partial_dependence_plot",
        "description": (
            "Generate a Partial Dependence Plot (PDP) for a feature. "
            "A PDP shows how the model's average predicted credit score changes as a feature varies, while averaging over all other features. "
            "Use this when the user asks about the global effect of a feature, e.g.: "
            "- 'How does a clean repayment history (On-time payment rate (%)) affect the score?' "
            "- 'What is the effect of recent applications (Months since last credit application) overall?' "
            " or use it when explaining the relationship between a feature and prediction globally."
        ),
        "parameters": PartialDependencePlot.model_json_schema(),
    }
]

