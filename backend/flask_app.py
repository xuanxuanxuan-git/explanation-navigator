import os
import json
import time
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
from openai import OpenAI, AzureOpenAI
from chatbot.explainers.tool_schemas import explainer_tools
from chatbot.explainers.tool_functions import available_tools_mapping

load_dotenv()

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TOOL_CALLING_MODEL = os.getenv("TOOL_CALLING_MODEL", "functiongemma:270m")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")]

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS)

# ============================================================================
# LOGGING SETUP - Using app.logger
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


# Configure app.logger
app.logger.setLevel(logging.DEBUG)

# Remove default handler
app.logger.handlers.clear()

# Make backend/logs/app.log relative to backend/app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # ensure directory exists
log_path = os.path.join(LOG_DIR, "app.log")

# File handler (writes to app.log)
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
app.logger.addHandler(file_handler)


# ============================================================================
# Endpoints support functions 
# ============================================================================


def _normalize_messages(messages):
    """
    Expect: [{ role: 'user'|'assistant'|'system'|'tool', content: '...' }, ...]
    """
    if not isinstance(messages, list):
        return []
    out = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("system", "user", "assistant", "tool") and content:
            out.append({"role": role, "content": content})
    return out


def get_tool_schemas():
    return explainer_tools

def _validate_tool_arguments(tool_name: str, arguments: dict) -> tuple:
    """
    Validate if tool arguments are complete.
    Returns: (is_valid: bool, missing_fields: list)
    """

    tool_schema = None
    for schema in get_tool_schemas():
        if schema.get("function", {}).get("name") == tool_name:
            tool_schema = schema
            break
    
    if not tool_schema:
        return False, ["Unknown tool"]
    
    required_fields = tool_schema.get("function", {}).get("parameters", {}).get("required", [])
    properties = tool_schema.get("function", {}).get("parameters", {}).get("properties", {})
    app.logger.info(properties)

    missing_fields = []
    for field in required_fields:
        if field not in arguments or arguments[field] is None:
            field_description = properties.get(field, {}).get("description", "")
            missing_fields.append({"name": field, "description": field_description})
            app.logger.info(field_description)
    # TODO: also check if the parameter type is correct
    is_valid = len(missing_fields) == 0

    return is_valid, missing_fields

def _execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool call and return the result"""

    is_valid, missing_fields = _validate_tool_arguments(tool_name, arguments)
    if not is_valid:
        field_names = [f['name'] for f in missing_fields]
        error_msg = f"Incomplete arguments for {tool_name}. Missing arguments: {', '.join(field_names)}"
        app.logger.warning(error_msg)
        
        missing_args_msg = "\n".join(f"- {f['name']}: {f['description']}" for f in missing_fields)  
        return False, missing_args_msg, None

    if tool_name not in available_tools_mapping:
        error_msg = f"Unknown tool: {tool_name}"
        app.logger.error(error_msg)
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
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Error executing {tool_name}: {str(e)}"
        return False, error_msg, None


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "time": int(time.time())})

@app.post("/api/chat")
def chat_non_stream():
    """
    Non-streaming chat: returns JSON { reply: "...", model: "..."}
    Uses Ollama POST /api/chat with stream:false.
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = _normalize_messages(data.get("history") or [])
    render_tools = data.get("render_tools", False)  # Flag to render tool call results; user_message is empty

    if not user_message and not render_tools:
        return jsonify({"error": "Missing 'message' field"}), 400

    # Set a system message here (or pass from frontend).
    system = (data.get("system") or "Answer questions succinctly.").strip()
    app.logger.info("pass here")
    try:
        messages = []
        if system and not render_tools:
            messages.append({"role": "system", "content": system})
        messages.extend(history)
        if not render_tools:
            messages.append({"role": "user", "content": user_message})
        app.logger.info(f"Chat history for api/chat: {messages}")
        payload = {
            "model": data.get("model") or OLLAMA_MODEL,
            "messages": messages,
            "stream": False,  # disable streaming
            "options": data.get("options") or {
                "temperature": 0.3
            }
        }

        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
        if r.status_code != 200:
            return jsonify({"error": "Ollama error", "status": r.status_code, "detail": r.text}), 502

        result = r.json()
        reply = (result.get("message") or {}).get("content") or ""
    
        return jsonify({"reply": reply, "model": payload["model"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 502


def _chat_with_tools(messages: str, model: str = None, #history: ,
                    options: dict = None, max_iterations: int = 1):
    """
    Chat with tool calling support. Handles tool calls iteratively.
    Returns: reply
    """
    tool_reply = []

    iteration = 0
    visualisations = []
    
    # TODO: why do we need multiple iterations?
    while iteration < max_iterations:
        iteration += 1
        
        # Call LLM with tools
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": get_tool_schemas(),  # Pass available tools
            "options": options
        }
        
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
        if r.status_code != 200:
            raise Exception(f"Ollama error: {r.status_code} - {r.text}")
        
        result = r.json()
        assistant_msg = result.get("message") or {}
        # structure of the assistant_msg: {"message": {"role": ..., "content": ..., "tool_calls": ...}}
        
        # Check for tool calls
        tool_calls = assistant_msg.get("tool_calls") or []
        
        if not tool_calls:
            # Add assistant response to tool_reply, and return
            tool_reply.append({"role": "assistant", "content": []})
            break
        
        # Execute each tool call
        # TODO: test that when multiple tool calls are detected, whether the reply is correctly formatted.
        for tool_call in tool_calls:
            func_info = tool_call.get("function") or {}
            tool_name = func_info.get("name")
            tool_args = func_info.get("arguments") or {}
            
            # Execute the tool
            success, tool_result, visualisation = _execute_tool_call(tool_name, tool_args)
            
            if not success:
                # Arguments are incomplete
                all_tools_valid = False
                app.logger.warning(f"Tool call validation failed: {tool_result}")
                
                # Add a message asking user for input
                user_input_msg = f"I need more information to call {tool_name}\n: {tool_result}"
                tool_reply.append({"role": "assistant", "content": user_input_msg})
                
            else:
                # Add tool result to messages
                tool_reply.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    # TODO: fix the following line so that when functions have no arguments, the content is adjusted.
                    "content": f"Tool {tool_name} called with arguments {tool_args}. Result: {tool_result}"
                })
                
                if visualisation:  # if there is visualisation result
                    visualisations.append({
                        "type": "plotly",
                        "figure": visualisation["figure"],
                        "config": visualisation.get("config", {}),
                        "meta": {
                            "tool": tool_name
                        }
                    })

    return tool_reply, visualisations


@app.post("/api/chat/tools")
def chat_with_tools():
    """
    Non-streaming chat endpoint with tool calling - will return tool execution results.
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = _normalize_messages(data.get("history") or [])
    system = (data.get("system") or "").strip()
    model = data.get("model") or TOOL_CALLING_MODEL
    
    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400
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
            options=data.get("options") or {"temperature": 0.5}
        )
        
        response = {
            "reply": reply,
            "model": model
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.post("/api/chat/tools/stream")
def chat_with_tools_stream():
    """
    Streaming chat endpoint with tool calling - will return tool execution results as a streaming response.
    """
    data = request.get_json(force=True) or {}    
    user_message = (data.get("message") or "").strip()
    history = _normalize_messages(data.get("history") or [])
    system = (data.get("system") or "").strip()
    model = data.get("model") or OLLAMA_MODEL
    tool_model = TOOL_CALLING_MODEL

    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    messages = []

    # Add system message if provided
    if system and not any(m.get("role") == "system" for m in history):
        messages.append({"role": "system", "content": system})

    # Add history
    messages.extend(history)
    app.logger.info(f"New chat starts. User question is: {user_message}")
    # app.logger.info(f"History: {history}")

    # Add user message
    messages.append({"role": "user", "content": user_message})
    app.logger.info(f"Messages passed to tool call: {messages}")
    try: 
        reply_with_tools, visualisations = _chat_with_tools(
            messages=messages,
            model=tool_model,
            options=data.get("options") or {"temperature": 0.3}
        )
        app.logger.info(f"Tool calling model: {tool_model}")
        app.logger.info(f"Reply from the tool call: {reply_with_tools}")
        messages.extend(reply_with_tools)
        
        messages_with_tools = _normalize_messages(messages)
        app.logger.info(f"Messages passed to streaming llm: {messages_with_tools}")

        payload = {
            "model": model,
            "messages": messages_with_tools,
            "render_tools": True,  
            "stream": True,  # streaming endpoint behaviour
            "options": data.get("options") or {
                "temperature": 0.2
            }
        }

        def generate():
            assistant_text = ""
            with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    yield f"event: error\ndata: {json.dumps({'status': r.status_code, 'detail': r.text})}\n\n"
                    return
                
                # Each line is a JSON object from Ollama.
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # incremental token chunk:
                    chunk = ((obj.get("message") or {}).get("content")) or ""
                    assistant_text += chunk
                    done = bool(obj.get("done"))
                    yield f"event: token\ndata: {json.dumps({'token': chunk, 'done': done})}\n\n"

                    if done:
                        messages_with_tools.append({"role": "assistant", "content": assistant_text})
                        app.logger.info(f"Completed response to stream: {messages_with_tools}")
                        yield f"event: visualisations\ndata: {json.dumps({'visualisations': visualisations})}\n\n"
                        yield f"event: done\ndata: {json.dumps({'done': True, 'history': messages_with_tools})}\n\n"
                        return

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.post("/api/chat/stream")
def chat_stream():
    """
    Streaming chat: returns Server-Sent Events (SSE).
    Ollama /api/chat streams JSON objects by default.
    """
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = _normalize_messages(data.get("history") or [])
    render_tools = data.get("render_tools", False)  # Flag to render tool call results; user_message is empty

    if not user_message and not render_tools:
        return jsonify({"error": "Missing 'message' field"}), 400

    system = (data.get("system") or "Answer questions succinctly.").strip()
    messages = []
    if system and not render_tools:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    if not render_tools:
        messages.append({"role": "user", "content": user_message})

    payload = {
        "model": data.get("model") or OLLAMA_MODEL,
        "messages": messages,
        "stream": True,  # streaming endpoint behaviour
        "options": data.get("options") or {
            "temperature": 0.2
        }
    }

    def generate():
        with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=300) as r:
            if r.status_code != 200:
                yield f"event: error\ndata: {json.dumps({'status': r.status_code, 'detail': r.text})}\n\n"
                return
            # Each line is a JSON object from Ollama.
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # incremental token chunk:
                chunk = ((obj.get("message") or {}).get("content")) or ""
                done = bool(obj.get("done"))
                yield f"event: token\ndata: {json.dumps({'token': chunk, 'done': done})}\n\n"

                if done:
                    yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"
                    return

    return Response(generate(), mimetype="text/event-stream")

@app.get("/api/tools")
def list_tools():
    """List available tools/functions"""
    return jsonify({"tools": get_tool_schemas()})

if __name__ == "__main__":
    app.logger.info("Starting Chatbot")
    app.logger.info(f"Host: {FLASK_HOST}, Port: {FLASK_PORT}")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
