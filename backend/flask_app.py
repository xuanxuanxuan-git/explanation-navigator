import os
import sys
import json
import time
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
from typing import Dict, List, Tuple, Optional
from openai import OpenAI, AzureOpenAI

from explainers.tool_schemas import explainer_tools
from explainers.tool_functions import available_tools_mapping, _STATE

load_dotenv()


# =============================================================================
# CONFIG
# =============================================================================

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))

# ---- Provider selection ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama | azure_openai | gemini

# ---- Ollama ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TOOL_CALLING_MODEL = os.getenv("TOOL_CALLING_MODEL", "functiongemma:270m")

# ---- Azure OpenAI ----
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")        # model to use for chat
AZURE_OPENAI_TOOL_DEPLOYMENT = os.getenv("AZURE_OPENAI_TOOL_DEPLOYMENT", AZURE_OPENAI_CHAT_DEPLOYMENT)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")]

app = Flask(__name__)
# CORS(app, origins=ALLOWED_ORIGINS)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
# ============================================================================
# LOGGING SETUP - Using app.logger
# ============================================================================

app.logger.setLevel(logging.DEBUG)
app.logger.handlers.clear()

# Use Azure's built-in LogFiles directory if running in Azure, otherwise use local 'logs' folder
if os.getenv("WEBSITE_SITE_NAME"):  # This variable always exists in Azure
    LOG_DIR = "/home/LogFiles/app_logs"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log_user_action(session_id, message):
    """Safely appends a log entry to a specific user's file."""
    if not session_id:
        session_id = "anonymous"
    app.logger.info(f"[{session_id}] {message}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = f"{session_id}.log"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - INFO - {message}\n")

# setup_app_logger()

# ============================================================================
# Endpoints support functions 
# ============================================================================

def _normalise_messages(messages):
    """
    Expect:
      system/user/assistant: {role, content}
      assistant tool-call:  {role, content, tool_calls}
      tool:                {role, tool_name, content}
    """
    if not isinstance(messages, list):
        return []

    out = []
    for m in messages:
        role = (m.get("role") or "").strip()
        if role not in ("system", "user", "assistant", "tool"):
            continue

        msg = {"role": role}

        # assistant content (optional when tool_calls exist)
        if "content" in m and m["content"] is not None:
            msg["content"] = str(m["content"])

        # preserve tool_calls
        if role == "assistant" and "tool_calls" in m:
            msg["tool_calls"] = m["tool_calls"]

        # preserve tool_name for tool messages
        if role == "tool":
            # msg["tool_name"] = m.get("tool_name")
            msg["tool_call_id"] = m.get("tool_call_id")

        # OpenAI requires either content or tool_calls
        if role == "assistant" and not msg.get("content") and not msg.get("tool_calls"):
            continue
        if role == "tool" and not msg.get("content"):
            continue

        out.append(msg)

    return out

def _shorten_messages(messages, num_tools):
    if not messages:
        return messages
    if messages[-1].get("role")=="assistant":
        messages.pop()
    
    recent_messages = messages[-num_tools:]

    for msg in recent_messages:
        if msg.get("role") != "tool":
            continue

        content = msg.get("content")

        try:
            payload = json.loads(content.replace("'", '"'))
        except Exception:
            continue

        if isinstance(payload, dict):
            payload.pop("sampled_values", None)
            payload.pop("prediction", None)
            payload.pop("trend_data", None)

            msg["content"] = json.dumps(payload)

    return messages

def get_tool_schemas(provider: Optional[str] = None):
    """
    Return tool schemas normalised for the target provider.

    Native schemas are OpenAI/Gemini-style:
      { "type": "function", "name": ..., "description": ..., "parameters": ... }

    Ollama and current OpenAI/Azure APIs expect:
      { "type": "function", "function": { "name": ..., "description": ..., "parameters": ... } }
    """
    provider = (provider or LLM_PROVIDER).lower()
    
    # Both Ollama and Azure OpenAI (especially newer models like gpt-5.1) 
    # require the nested "function" structure.
    needs_wrapped = provider in ["ollama", "azure_openai", "openai", "azure"]
    
    if needs_wrapped:
        normalised = []
        for t in explainer_tools:
            # Accept both already-wrapped and flat schemas
            if "function" in t:
                normalised.append(t)
            else:
                normalised.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("parameters"),
                    },
                })
        return normalised

    # Fallback for Gemini which accepts the flat schema
    return explainer_tools

def _validate_tool_arguments(tool_name: str, arguments: dict) -> tuple:
    """
    Validate if tool arguments are complete.
    Returns: (is_valid: bool, missing_fields: list)
    """

    tool_schema = None
    for schema in explainer_tools:
        name = schema.get("name") or schema.get("function", {}).get("name")
        if name == tool_name:
            tool_schema = schema
            break
    
    if not tool_schema:
        return False, [{"name": "tool_name", "description": f"Unknown tool: {tool_name}"}]

    parameters = tool_schema.get("parameters") or tool_schema.get("function", {}).get("parameters", {})
    required_fields = parameters.get("required", [])
    properties = parameters.get("properties", {})

    missing_fields = []
    for field in required_fields:
        if field not in arguments or arguments[field] is None:
            field_description = properties.get(field, {}).get("description", "")
            missing_fields.append({"name": field, "description": field_description})

    # TODO: also check if the parameter type is correct
    is_valid = len(missing_fields) == 0

    return is_valid, missing_fields


def _execute_tool_call(tool_name: str, arguments: dict, session_id: str = "anonymous") -> Tuple[bool, str, Optional[dict]]:
    """Execute a tool call and return (success, text, visualisation)"""

    is_valid, missing_fields = _validate_tool_arguments(tool_name, arguments)
    if not is_valid:
        field_names = [f['name'] for f in missing_fields]
        error_msg = f"Incomplete arguments for {tool_name}. Missing arguments: {', '.join(field_names)}"
        # app.logger.warning(error_msg)
        log_user_action(session_id, f"TOOL VALIDATION FAILED: {error_msg}")
        
        missing_args_msg = "\n".join(f"- {f['name']}: {f['description']}" for f in missing_fields)  
        return False, missing_args_msg, None

    if tool_name not in available_tools_mapping:
        error_msg = f"Unknown tool: {tool_name}"
        # app.logger.error(error_msg)
        log_user_action(session_id, f"TOOL ERROR: {error_msg}")
        return False, error_msg, None
    
    try:
        func = available_tools_mapping[tool_name]
        result = func(**arguments)
        
        # If result is a dict with 'data' and 'visualisation', split them
        visualisation_data = None
        result_text = str(result)
        
        if isinstance(result, dict):
            result_text = result.get("data", str(result))
            visualisation_data = result.get("visualisation")
        
        return True, result_text, visualisation_data

    except TypeError as e:
        error_msg = f"Error calling {tool_name}: {str(e)}"
        log_user_action(session_id, f"TOOL CRASH: {error_msg}")
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Error executing {tool_name}: {str(e)}"
        log_user_action(session_id, f"TOOL CRASH: {error_msg}")
        return False, error_msg, None

# =============================================================================
# LLM CLIENT ABSTRACTION
# =============================================================================

class BaseLLMClient:
    """Unified interface for chat + streaming + tool calling."""

    def chat(self, messages: List[dict], model: Optional[str] = None,
             options: Optional[dict] = None,
             tools: Optional[list] = None) -> dict:
        """Return a single assistant message in OpenAI-style format."""
        raise NotImplementedError

    def stream(self, messages: List[dict], model: Optional[str] = None,
               options: Optional[dict] = None,
               tools: Optional[list] = None):
        """Yield incremental assistant tokens (strings)."""
        raise NotImplementedError

# -----------------------------------------------------------------------------
# OLLAMA CLIENT
# -----------------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(self, messages, model=None, options=None, tools=None):
        payload = {
            "model": model or OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": options or {},
        }
        if tools:
            payload["tools"] = tools

        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("message") or {}

    def stream(self, messages, model=None, options=None, tools=None):
        payload = {
            "model": model or OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": options or {},
        }
        if tools:
            payload["tools"] = tools

        with requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = ((obj.get("message") or {}).get("content")) or ""
                done = bool(obj.get("done"))
                yield chunk, done


# -----------------------------------------------------------------------------
# AZURE OPENAI CLIENT
# -----------------------------------------------------------------------------

class AzureOpenAIClient(BaseLLMClient):
    def __init__(self, api_key, endpoint, api_version):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def chat(self, messages, model=None, options=None, tools=None):
        response = self.client.chat.completions.create(
            model=model or AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages= messages, #self._to_openai_messages(messages),
            tools=tools,
            temperature=(options or {}).get("temperature", 1),
        )

        msg = response.choices[0].message
        return msg.model_dump()

    def stream(self, messages, model=None, options=None, tools=None):

        stream = self.client.chat.completions.create(
            model=model or AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages= messages, #self._to_openai_messages(messages),
            tools=tools,
            temperature=(options or {}).get("temperature", 1),
            stream=True,
        )

        for event in stream:
            if event.choices and event.choices[0].delta:
                delta = event.choices[0].delta
                token = delta.content or ""
                yield token, False
        yield "", True


# =============================================================================
# CLIENT FACTORY
# =============================================================================

def get_llm_client(provider: str) -> BaseLLMClient:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(OLLAMA_BASE_URL)
    elif provider in ("azure_openai", "openai", "azure"):
        return AzureOpenAIClient(
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    elif provider == "gemini":
        return GeminiClient(GEMINI_API_KEY)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


llm_client = get_llm_client(LLM_PROVIDER)


# =============================================================================
# TOOL-CALLING LOOP (LLM-AGNOSTIC)
# =============================================================================

def _chat_with_tools(messages: list, model: str = None, #history: ,
                    options: dict = None, max_iterations: int = 3, session_id: str = "anonymous"):
    """
    Chat with tool calling support. Handles tool calls iteratively.
    Returns: (tool_reply_messages, visualisations)
    """
    local_messages = list(messages)     # create a local copy
    tool_reply = []

    iteration = 0
    visualisations = []
    
    while iteration < max_iterations:
        # app.logger.info(f"Tool iteration {iteration}")
        log_user_action(session_id, f"Tool iteration {iteration}")
        iteration += 1
        
        assistant_msg = llm_client.chat(
            messages=local_messages,
            model=model,
            options=options,
            tools=get_tool_schemas(LLM_PROVIDER),
        )
        # structure of the assistant_msg: {"role": ..., "content": ..., "tool_calls": ...}
        
        # Check for tool calls
        tool_calls = assistant_msg.get("tool_calls") or []
        # app.logger.info(f"Tools called: {tool_calls}")
        log_user_action(session_id, f"Tools called: {tool_calls}")
        if not tool_calls:
            # Add assistant response to tool_reply, and return
            tool_reply.append({
                "role": "assistant",
                "content": assistant_msg.get("content", ""), #[]
            })
            break
        else:
            # in case more iterations might be needed in the last iteration, run it one more time
            if iteration == max_iterations:
                max_iterations += 1
        
        # Append assistant tool-call message 
        assistant_tool_msg = {
            "role": "assistant",
            "content": assistant_msg.get("content", ""),
            "tool_calls": tool_calls,
        }

        tool_reply.append(assistant_tool_msg)
        local_messages.append(assistant_tool_msg)

        # Execute each tool call
        for tool_call in tool_calls:
            func_info = tool_call.get("function") or {}
            tool_name = func_info.get("name")
            raw_args = func_info.get("arguments") or {}

            # some LLM returns arguments as a string, not a dict
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    tool_args = {}
            else:
                tool_args = raw_args

            # Execute the tool
            success, tool_result, visualisation = _execute_tool_call(tool_name, tool_args, session_id)
            
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": (
                    f"{tool_result}"
                    if success                      # Add tool result to messages
                    else f"ERROR: {tool_result}"    # Add a message asking user for input
                ),
            }

            if tool_name == "get_subgroup":
                shorten_tool_msg = tool_msg.copy()
                content = shorten_tool_msg.get("content")
                try:
                    payload = json.loads(content.replace("'", '"'))

                    if isinstance(payload, dict):
                        payload.pop("indices", None)
                        shorten_tool_msg["content"] = json.dumps(payload) 
                except Exception:
                    pass

                tool_reply.append(shorten_tool_msg)
            else:
                tool_reply.append(tool_msg)
            # app.logger.info(f"Tool implementation result is: {tool_msg}")
            log_user_action(session_id, f"Tool implementation result is: {tool_msg}")
            local_messages.append(tool_msg)
            
            if not success:
                # Arguments are incomplete
                app.logger.warning(f"Tool call validation failed: {tool_result}")

                # Stop loop so model can ask user for missing arguments
                break
            if visualisation:  # if there is visualisation result
                visualisations.append({
                    "type": "plotly",
                    "figure": visualisation["figure"],
                    "config": visualisation.get("config", {}),
                    "meta": visualisation.get("meta", {}) 
                })

    return tool_reply, visualisations

# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "time": int(time.time()), "provider": LLM_PROVIDER})

@app.get("/api/session/reset")
def reset_session():
    return jsonify({"message": "Session reset successfully"}), 200  
    
@app.post("/api/chat")
def chat_non_stream():
    """
    Non-streaming chat: returns JSON { reply: "...", model: "..."}
    LLM-provider agnostic with stream=False.
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = _normalise_messages(data.get("history") or [])
    render_tools = data.get("render_tools", False)  # Flag to render tool call results; user_message is empty
    session_id = data.get("session_id", "anonymous")

    if not user_message and not render_tools:
        return jsonify({"error": "Missing 'message' field"}), 400

    if user_message:
        log_user_action(session_id, f"USER ASKED: {user_message}")

    # Set a system message here (or pass from frontend).
    system = (data.get("system") or "Answer questions succinctly.").strip()
    
    try:
        messages = []
        if system and not render_tools:
            messages.append({"role": "system", "content": system})
        messages.extend(history)
        if not render_tools:
            messages.append({"role": "user", "content": user_message})
        # app.logger.info(f"Chat history for api/chat: {messages}")
        log_user_action(session_id, f"Chat history for api/chat: {messages}")
        
        assistant_msg = llm_client.chat(
            messages=messages,
            model=data.get("model"),
            options=data.get("options") or {"temperature": 1},
        )

        reply = assistant_msg.get("content", "")
        log_user_action(session_id, f"LLM REPLIED: {reply}")
        return jsonify({"reply": reply, "model": data.get("model")})

    except Exception as e:
        log_user_action(session_id, f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 502


@app.post("/api/chat/tools")
def chat_with_tools():
    """
    Non-streaming chat endpoint with tool calling - will return tool execution results.
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = _normalise_messages(data.get("history") or [])
    system = (data.get("system") or "").strip()
    model = data.get("model")
    session_id = data.get("session_id", "anonymous")
    
    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400
        
    log_user_action(session_id, f"USER ASKED: {user_message}")
        
    try:
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add history
        messages.extend(history)

        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Call with tools
        reply, _ = _chat_with_tools(
            messages=messages,
            model=model,
            options=data.get("options") or {"temperature": 1},
            session_id=session_id
        )

        return jsonify({"reply": reply, "model": model})

    except Exception as e:
        log_user_action(session_id, f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 502


@app.post("/api/chat/tools/stream")
def chat_with_tools_stream():
    """
    Streaming chat endpoint with tool calling - will return tool execution results as a streaming response.
    """
    data = request.get_json(force=True) or {}    
    user_message = (data.get("message") or "").strip()
    history = _normalise_messages(data.get("history") or [])
    system = (data.get("system") or "").strip()
    model = data.get("model")
    session_id = data.get("session_id", "anonymous")

    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400
        
    log_user_action(session_id, f"New chat starts. User question is: {user_message}")
    
    messages = []

    # Add system message if provided
    if system and not any(m.get("role") == "system" for m in history):
        messages.append({"role": "system", "content": system})

    # Add history
    messages.extend(history)

    # Add user message
    messages.append({"role": "user", "content": user_message})
    log_user_action(session_id, f"Messages passed to tool call: {messages}")
    
    try: 
        reply_with_tools, visualisations = _chat_with_tools(
            messages=messages,
            model=model,
            options=data.get("options") or {"temperature": 1},
            session_id=session_id
        )

        log_user_action(session_id, f"Reply from the tool call: {reply_with_tools}")
        
        num_tools = len(reply_with_tools)-1
        messages.extend(reply_with_tools)
        
        messages_with_tools = _normalise_messages(messages)
        log_user_action(session_id, f"Messages passed to streaming llm: {messages_with_tools}")

        payload_model = data.get("final_model") or data.get("model")

        def generate():
            try:
                assistant_text = ""
                for chunk, done in llm_client.stream(
                    messages=messages_with_tools,
                    model=payload_model,
                    options={"temperature": 0.1},
                ):
                    assistant_text += chunk
                    yield f"event: token\ndata: {json.dumps({'token': chunk, 'done': done})}\n\n"

                    if done:
                        _shorten_messages(messages_with_tools, num_tools)
                        messages_with_tools.append({"role": "assistant", "content": assistant_text})
                        log_user_action(session_id, f"Complete message saved: {messages_with_tools}")
                        
                        yield f"event: visualisations\ndata: {json.dumps({'visualisations': visualisations})}\n\n"
                        yield f"event: done\ndata: {json.dumps({'done': True, 'history': messages_with_tools})}\n\n"
                        return
            except Exception as e:
                # IMPORTANT: If LLM fails during stream, log it but emit a done event to unlock the UI
                log_user_action(session_id, f"STREAMING ERROR: {str(e)}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                yield f"event: done\ndata: {json.dumps({'done': True, 'history': messages_with_tools})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        log_user_action(session_id, f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 502


@app.post("/api/chat/stream")
def chat_stream():
    """
    Streaming chat: returns Server-Sent Events (SSE).
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = _normalise_messages(data.get("history") or [])
    render_tools = data.get("render_tools", False)  # Flag to render tool call results; user_message is empty
    session_id = data.get("session_id", "anonymous")

    if not user_message and not render_tools:
        return jsonify({"error": "Missing 'message' field"}), 400

    if user_message:
        log_user_action(session_id, f"USER ASKED: {user_message}")

    system = (data.get("system") or "Answer questions succinctly.").strip()
    messages = []
    if system and not render_tools:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    if not render_tools:
        messages.append({"role": "user", "content": user_message})

    # payload = {
    #     "model": data.get("model") or OLLAMA_MODEL,
    #     "messages": messages,
    #     "stream": True,  # streaming endpoint behaviour
    #     "options": data.get("options") or {
    #         "temperature": 0.2
    #     }
    # }

    def generate():
        try:
            assistant_text = ""
            for chunk, done in llm_client.stream(
                messages=messages,
                model=data.get("model"),
                options=data.get("options") or {"temperature": 1},
            ):
                assistant_text += chunk
                yield f"event: token\ndata: {json.dumps({'token': chunk, 'done': done})}\n\n"

                if done:
                    log_user_action(session_id, f"LLM REPLIED: {assistant_text}")
                    yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"
                    return
        except Exception as e:
            # IMPORTANT: Unlock UI on stream failure
            log_user_action(session_id, f"STREAMING ERROR: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.get("/api/tools")
def list_tools():
    """List available tools/functions"""
    return jsonify({"tools": get_tool_schemas(LLM_PROVIDER)})


@app.get("/api/instance/<int:instance_id>")
def get_instance(instance_id):
    try:
        profile = available_tools_mapping["get_instance_features_and_prediction"](instance_id=instance_id)
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/instance/predict")
def predict_instance_with_changes():
    data = request.get_json(force=True) or {}
    instance_id = data.get("instance_id")
    changes = data.get("changes") or {}

    if instance_id is None:
        return jsonify({"error": "Missing instance_id"}), 400

    try:
        result = available_tools_mapping["predict_with_feature_changes"](
            instance_id=int(instance_id),
            changes=changes,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/instance/<int:instance_id>/shap-bar-plot")
def shap_bar_plot(instance_id):
    try:
        max_display = request.args.get("max_display", default=10, type=int)
        result = available_tools_mapping["generate_local_shap_bar_plot"](
            instance_id=instance_id,
            max_display=max_display,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/instance/<int:instance_id>/counterfactual-explanation")
def counterfactual_explanation(instance_id):
    try:
        max_steps = request.args.get("max_steps", default=50, type=int)
        target = request.args.get("target", default=None, type=float)
        result = available_tools_mapping["get_counterfactual_explanation"](
            instance_id=instance_id,
            target=target,
            max_steps=max_steps,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/global/global-shap-plot")
def global_shap_plot():
    try:
        max_display = request.args.get("max_display", default=10, type=int)
        result = available_tools_mapping["generate_global_subgroup_shap_plot"](
            source="all",
            indices=None,
            max_display=max_display,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/instance/<int:instance_id>/cp-plots")
def generate_all_cp_plots(instance_id):
    """
    Endpoint to retrieve CP plots for all features.
    """
    try:
        result = available_tools_mapping["generate_all_cp_plots"](
            instance_id=instance_id,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting Chatbot")
    print(f"Host: {FLASK_HOST}, Port: {FLASK_PORT}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)