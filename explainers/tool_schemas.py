# Define tool schemas
from typing import Dict, Union, List, Optional
from pydantic import BaseModel, Field

# define the tool properties with pydantic
class ShapBarPlot(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based). Returns the contributions of factors that pushed this person's credit score up or down.")

class ShapSummaryPlot(BaseModel):
    source: str = Field(
        default="all",
        description="Use 'all' to see which factors matter across everyone, or 'indices' to look at a specific group of applicants"
    )
    indices: Optional[List[int]] = Field(
        default=None,
        description="List of applicant indices when source='indices'. Usually obtained from get_subgroup."
    )

class IndividualPrediction(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based). Returns their profile details and predicted credit score.")

class AveragePrediction(BaseModel):
    source: str = Field(
        default="all",
        description="Use 'all' to average over all applicants, or 'indices' to average over a selected subgroup."
    )
    indices: Optional[List[int]] = Field(
        default=None,
        description="List of applicant indices to include when source='indices'."
    )

class CpPlot(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based).")
    feature: str = Field(description="Exact factor to vary (e.g. Credit used (%), Months since last late payment, On-time payment rate (%)).")

class Condition(BaseModel):
    op: str = Field(
        description="Comparison operator. One of: >, >=, <, <=, ==, gt, ge, lt, le, eq"
    )
    value: float = Field(description="Numeric value to compare against.")

class Subgroup(BaseModel):
    filters: Dict[str, Union[Condition, List[Condition]]] = Field(
        description=(
            "Map of factor name to filtering condition(s). "
            "Each factor can have a single condition or a list of conditions. "
            "If multiple conditions are provided, all must be satisfied (logical AND)."
        )
    )

class PredictWithFeatureChanges(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based).")
    changes: Dict[str, float] = Field(
        description=(
            "Map of a factor name to new numeric value. Examples: {'Number of loans': 80} if changing Number of loans to 80."
        )
    )

class CounterfactualExplanation(BaseModel):
    instance_id: int = Field(description="Index of the applicant (0-based).")
    target: Optional[float] = Field(
        default=None,
        description="Desired credit score (0-100, >=50 means approved)."
    )

class SimilarInstances(BaseModel):
    instance_id: int = Field(
        description="Index of the query applicant (0-based) to find similar people for."
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
            "Factor name to analyse. Must match an existing factor exactly."
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
    feature: str = Field(description="Exact feature name to see how it affects credit scores globally across all applicants.")


explainer_tools = [
    {
        "type": "function",
        "name": "generate_local_shap_bar_plot",
        "description": "Shows what factors pushed ONE applicant's predicted score up or down. "
        "Use this when the user asks things like 'why is my score high/low?' or 'what affected my score the most?'. "
        "This explanation applies only to the selected applicant and does NOT represent "
        "factor importance across the dataset.", 
        "parameters": ShapBarPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "generate_global_subgroup_shap_plot",
        "description": """
            Shows which factors matter the most on average, or across a specific group of people. 
            Use this when the user asks questions such as:
            - "Which factors are most important in the model?"
            - "What factors matter most overall?"
            - "For applicants with more than 12 months since last late payment, which factors influence their predicted credit score the most?"
            Do NOT used this tool for questions about a specific person.", 
        """,
        "parameters": ShapSummaryPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_instance_features_and_prediction",
        "description": (
            "Returns the factor values and predicted credit score for a SINGLE applicant. "
            
            "Use this when the user asks about one applicant's data, such as: "
            "- 'what are my factor values?' "
            "- 'what is my predicted credit score?' " 
            "Also use it internally when you need to know their current values before explaining things."
        ),  
        "parameters": IndividualPrediction.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_average_prediction",
        "description": "Compute average predicted credit score over all applicants or over a provided list of applicant indices (for example, indices returned by get_subgroup).",
        "parameters": AveragePrediction.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_cp_plot",
        "description": "Shows how one applicant's predicted credit score changes when changing a single factor while all other factors are kept fixed at their original values. "
        "Use this tool when the user asks:"
            "- 'How does this factor affect my score?'"
            "- 'What happens if I lower my Number of loans'"
            "- 'What if this factor was higher or lower?'"
        "Note: This explanation only applies for a specific applicant, and only ONE factor is varied at a time. Feature must match a valid factor name exactly.",
        "parameters": CpPlot.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_subgroup",
        "description": """
            Filter applicants by feature conditions and return the indices of applicants who satisfy the conditions.

            Each filter specifies a factor name and a comparison condition. The factor name must match a feature exactly. You may also filter on 'prediction'. 

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
            Alter one or more factor values for a single applicant and return the new predicted credit score. Use this for what-if analysis such as changing an applicant's financial attributes. 
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
            "Return representative applicants of a subgroup. "
            "The subgroup must first be identified using the get_subgroup tool, "
            "which returns a list of indices. These indices should then be passed "
            "to this function. The function returns typical examples of that group."
        ),
        "parameters": RepresentativeInstances.model_json_schema(),
    },
    {
        "type": "function",
        "name": "dataset_meta",
        "description": (
            "Provide information about the credit scoring dataset used by the model. "
            "Use this when the user asks about the dataset overview (size, features, target), "
            "summary statistics, or the distribution of a specific feature. "
            "If users want to see the general spread of a factor, pass in 'feature' name. "
            "If users ask 'where do I stand compared to everyone else?' pass both 'feature' and 'instance_id' in."
        ),
        "parameters": DatasetMeta.model_json_schema(),
    },
    {
        "type": "function",
        "name": "model_meta",
        "description": (
            "Provides general information about how the AI system works, its accuracy, and what it tries to predict."
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
            "Calculates the smallest combination of changes a user can make to improve their credit score to a target value. "
            "Emphasise that this represents the MINIMUM required changes, but these specific changes must be implemented simultaneously to reach the target score. Individual changes should NOT be interpreted in isolation. "
            "Only include `target` if the user explicitly specifies a desired score."
            "Use this tool when the user asks: "
            "- 'How can I get a higher score' -> {'instance_id': }"
            "- 'How can I increase the predicted score to 60?' -> {'instance_id': , 'target': 60}"
            "- 'What is the easiest way to improve my score?' -> {'instance_id': }"
        ),
        "parameters": CounterfactualExplanation.model_json_schema(),
    },
    {
        "type": "function",
        "name": "get_partial_dependence_plot",
        "description": (
            "Shows the general trend of how a specific factor affects scores across everyone in the system. "
            "Use this when the user asks about the average effect of a feature, e.g.: "
            "- 'How does a clean repayment history (On-time payment rate (%)) affect the score overall?' "
            "- 'What is the effect of recent applications (Months since last credit application) on average?' "
            "- 'Do more loans generally lower the score?'"
        ),
        "parameters": PartialDependencePlot.model_json_schema(),
    }
]

