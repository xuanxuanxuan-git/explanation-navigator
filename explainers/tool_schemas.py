# Define tool schemas

from pydantic import BaseModel, Field

# define the tool properties with pydantic
class ShapBarPlot(BaseModel):
    instance_id: int = Field(
        default=0,
        description="The index of the instance to generate SHAP bar plot for"
    )

explainer_tools = [
    # First tool -- generate bar chart for local feature attribution
    {
        "type": "function",
        "name": "generate_shap_bar_plot",
        "description": "Generates a bar chart showing SHAP values for local feature attribution of a specific model prediction. Use this when the user wants to understand the importance of features for a particular instance's prediction.",
        "parameters": ShapBarPlot.model_json_schema(),
    },
    # Second tool -- generate summary plot for global feature importance
    {
        "type": "function",
        "name": "generate_shap_summary_plot",
        "description": "Generates a summary plot showing global feature importance across the entire dataset using SHAP values. Use this to visualize which features are most important on average for model predictions, not for a specific instance.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # third tool -- return the shap value of each feature for a specific instance
    {
        "type": "function",
        "name": "get_feature_attribution_ranking",
        "description": "Returns the SHAP-based local feature attribution ranking for a specific instance.",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "integer",
                    "description": "Index of the instance to analyze."
                }
            },
            "required": ["instance_id"],
        },
    },
    {
        "type": "function",
        "name": "predict_price_probability",
        "description": "Estimates how likely houses in this dataset are going to have price over $30k by averaging the model predicted price over all data instances. This provides a population-level price estimate, not an explanation for any specific house.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

### Ollama style
# explainer_tools = [
#     # First tool -- generate bar chart for local feature attribution
#     {
#         "type":"function",
#         "function": {
#             "name": "generate_shap_bar_plot",
#             "description": "Generates a bar chart showing SHAP values for local feature attribution of a specific model prediction. Use this when the user wants to understand which features is the most important for a particular instance's prediction.",
#             "parameters": ShapBarPlot.model_json_schema(),
#             },
#         },
#     # Second tool -- generate summary plot for global feature importance
#     {
#         "type":"function",
#         "function": {
#             "name": "generate_shap_summary_plot",
#             "description": "Generates a summary plot showing global feature importance across the entire dataset using SHAP values. Use this to visualize which features are most important on average for model predictions, not for a specific instance.",
#             "parameters": {
#                 "type":"object",
#                 "properties": {},
#                 "required":[],
#             },
#         },
#     },
#     # third tool -- return the shap value of each feature for a specific instance
#     {
#         "type": "function",
#         "function": {
#             "name": "get_feature_attribution_ranking",
#             "description": "Returns the SHAP-based local feature attribution ranking for a specific instance, sorted by absolute importance. Use this when a user asks which feature was most influential for a particular prediction.",
#             "parameters": {
#             "type": "object",
#             "properties": {
#                 "instance_id": {
#                     "type": "integer",
#                     "description": "Index of the instance to analyze for local feature importance."
#                 }
#             },
#             "required": ["instance_id"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "predict_price_probability",
#             "description": "Estimates how likely houses in this dataset are going to have price over $30k by averaging the model predicted price over all data instances. This provides a population-level price estimate, not an explanation for any specific house.",
#             "parameters": {
#                 "type":"object",
#                 "properties": {},
#                 "required":[],
#             }, 
#         }
#     },
# ]

