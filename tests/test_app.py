import os
import requests
import json
from dotenv import load_dotenv
from time import sleep

load_dotenv()

# Backend URL
# BASE_URL = int(os.getenv("FLASK_PORT", "5001"))
BASE_URL = "http://localhost:5001"

def test_health():
    """Test health endpoint"""
    print("\n === Testing /api/health ===")
    try:
        r = requests.get(f"{BASE_URL}/api/health")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
        assert r.status_code == 200
        assert r.json()['ok'] is True
        print("Health check passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_simple_chat():
    """Test simple non-streaming one-time chat"""
    print("\n=== Testing /api/chat (non-streaming) ===")

    simple_message = "How does counterfactual explanation work?"
    try:
        payload = {
            "message": simple_message,
            # "model": "llama",
        }
        print(f"Sending message: {payload}")
        r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=120)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
        assert r.status_code==200
        
        data = r.json()
        assert 'reply' in data
        assert len(data['reply']) > 0
        print("Chat test passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False



def test_chat_with_history():
    """Test chat with conversation history"""
    print("\n=== Testing /api/chat with history ===")
    try:
        # First message
        payload1 = {
            "message": "Tell me about counterfactual explanations used in AI.",
            "history": [],
        }
        print(f"First message: {payload1['message']}")
        r1 = requests.post(f"{BASE_URL}/api/chat", json=payload1, timeout=120)
        assert r1.status_code == 200
        reply1 = r1.json()['reply']
        print(f"Assistant: {reply1[:200]}...")

        # Second message with history
        history = [
            {"role": "user", "content": payload1["message"]},
            {"role": "assistant", "content": reply1}
        ]
        payload2 = {
            "message": "Explain it in one short sentence.",
            "history": history,
        }
        print(f"Second message: {payload2['message']}")
        r2 = requests.post(f"{BASE_URL}/api/chat", json=payload2, timeout=120)
        assert r2.status_code == 200
        
        reply2 = r2.json()['reply']
        print(f"Assistant: {reply2[:200]}...")
        print("History test passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_chat_with_system_prompt():
    """Test chat with custom system prompt"""
    print("\n=== Testing /api/chat with system prompt ===")
    try:
        payload = {
            "message": "Tell me about counterfactual explanations used in AI.",
            "history": [],
            "system": "Assume users have no AI knowledge.",
        }
        print(f"System prompt: {payload['system']}")
        print(f"User message: {payload['message']}")
        r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=120)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Assistant: {data['reply'][:150]}...")
        assert r.status_code == 200
        assert len(data['reply']) > 0
        print("System prompt test passed")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_chat_with_options():
    """Test chat with custom temperature option"""
    print("\n=== Testing /api/chat with custom options ===")
    try:
        payload = {
            "message": "Tell me about counterfactual explanations used in AI.",
            "history": [],
            "options": {"temperature": 0.8},  # Higher temperature = more creative
        }
        print(f"Options: {payload['options']}")
        print(f"User message: {payload['message']}")
        r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=120)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Assistant: {data['reply'][:150]}...")
        assert r.status_code == 200
        print("Options test passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_chat_streaming():
    """Test streaming chat endpoint"""
    print("\n=== Testing /api/chat/stream (streaming) ===")
    try:
        payload = {
            "message": "What is counterfactual explanation in AI?",
            "history": [],
        }
        print(f"User message: {payload['message']}")
        r = requests.post(
            f"{BASE_URL}/api/chat/stream",
            json=payload,
            stream=True,
            timeout=120
        )
        print(f"Status: {r.status_code}")
        assert r.status_code == 200

        print("Streaming response:")
        full_response = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                # print("\nevent is: ", event)
                if event in ("done"):
                    print(f"  [{event}]", end=" ")
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(data_str)
                    if "token" in data:
                        token = data["token"]
                        full_response += token
                        print(token, end="", flush=True)
                except json.JSONDecodeError:
                    pass

        print("\nStreaming test passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_error_missing_message():
    """Test error handling for missing message"""
    print("\n=== Testing error: missing message ===")
    try:
        payload = {
            "history": [],
            # "message" is missing
        }
        r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=10)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
        assert r.status_code == 400
        assert 'error' in r.json()
        print("Error handling test passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_empty_message():
    """Test error handling for empty message"""
    print("\n=== Testing error: empty message ===")
    try:
        payload = {
            "message": "   ",  # just whitespace
            "history": [],
        }
        r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=10)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
        assert r.status_code == 400
        print("Empty message error handling passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


### --------------- Tool call tests -------------------- ###

def test_list_tools():
    """Test listing available tools"""
    print("\n=== Testing /api/tools ===")
    try:
        r = requests.get(f"{BASE_URL}/api/tools")
        print(f"Status: {r.status_code}")
        data = r.json()
        tools = data['tools']
        print(f"Available tools: {len(tools)}")
        for tool in tools:
            func = tool.get('function') or {}
            print(f"  - {func.get('name')}: {func.get('description')}")
        assert r.status_code == 200
        # assert len(tools) >= 3  # get_weather, calculate_math, get_current_time
        print("Tools endpoint passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_tool_calling_simple():
    """Test single tool calling"""
    print("\n=== Testing /api/chat with tool calling ===")
    try:
        payload1 = {
            "system": "You have access to tools but they are optional.",
            # "message": "how likely is a house to have a price over $30k?",
            "message": "What is the most important feature for instance 21?",
            # "message": "Which country is London in?",
            "history": [],
            # "model": "llama3.2:3b",
        }
        print(f"User message: {payload1['message']}")
        r1 = requests.post(f"{BASE_URL}/api/chat/tools", json=payload1, timeout=180)
        print(f"Tool reply status: {r1.status_code}")
        assert r1.status_code == 200
        tool_reply = r1.json()
        print(f"Assistant reply: {tool_reply['reply']}")
        print(f"Model: {tool_reply['model']}" )

        history = [
            {"role": "system", "content": "Answer questions succinctly using the answer returned by the tool. "
                                        "Only when no results are returned from tool calls, answer yourself."}, 
            {"role":"user", "content": payload1["message"]},
        ]
        history.extend(tool_reply['reply'])
        print("Chat history:", history)
        payload2 = {
            "history": history,
            "render_tools": True  # render responses from tool calls
        }
        
        r2 = requests.post(f"{BASE_URL}/api/chat", json=payload2, timeout=180)
        print(f"Final reply status: {r2.status_code}")
        assert r2.status_code == 200
        reply2 = r2.json()["reply"]
        print(f"Final response: {reply2}")
        print(f"Model: {r2.json()['model']}" )

        print("Simple tool calling passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_tool_calling_stream():
    """Test tool calling and streaming final response"""
    print("\n=== Testing /api/chat with tool calling ===")
    try:
        payload1 = {
            "system": "You have access to tools but they are optional.",
            # "message": "how likely is a house to have a price over $30k?",
            "message": "What is the most important feature for instance 21?",
            # "message": "Which country is London in?",
            "history": [],
            # "model": "llama3.2:3b",
        }
        print(f"User message: {payload1['message']}")
        r1 = requests.post(f"{BASE_URL}/api/chat/tools", json=payload1, timeout=180)
        print(f"Tool reply status: {r1.status_code}")
        assert r1.status_code == 200
        tool_reply = r1.json()
        print(f"Assistant reply: {tool_reply['reply']}")
        print(f"Model: {tool_reply['model']}" )

        history = [
            {"role": "system", "content": "Answer questions succinctly using the answer returned by the tool. "
                                        "Only when no results are returned from tool calls, answer yourself."}, 
            {"role":"user", "content": payload1["message"]},
        ]
        history.extend(tool_reply['reply'])
        print("Chat history:", history)
        payload2 = {
            "history": history,
            "render_tools": True,  # render responses from tool calls
        }
        
        r2 = requests.post(f"{BASE_URL}/api/chat/stream", json=payload2, stream=True, timeout=180)
        print(f"Final reply status: {r2.status_code}")
        assert r2.status_code == 200
        
        print("Streaming response:")
        full_response = ""
        for line in r2.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                if event in ("done"):
                    print(f"  [{event}]", end=" ")
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(data_str)
                    if "token" in data:
                        token = data["token"]
                        full_response += token
                        print(token, end="", flush=True)
                except json.JSONDecodeError:
                    pass

        print("\nStreaming tool calling response passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat_with_tools_streaming():
    """Test chat with tools and streaming final response"""
    print("\n=== Testing /api/chat/tools/stream ===")
    try:
        payload = {
            "system": "Answer questions succinctly using the answer returned by the tool.",
            "message": "What is the most important feature for instance 21?",
            "history": [],
        }
        print(f"User message: {payload['message']}")
        r = requests.post(
            f"{BASE_URL}/api/chat/tools/stream",
            json=payload,
            # stream=True,
            timeout=180
        )
        print(f"Status: {r.status_code}")
        assert r.status_code == 200

        print("Streaming response with tool results:")
        full_response = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
                if event in ("done"):
                    print(f"  [{event}]", end=" ")
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(data_str)
                    if "token" in data:
                        token = data["token"]
                        full_response += token
                        print(token, end="", flush=True)
                except json.JSONDecodeError:
                    pass

        print("\nTool calling with streaming test passed")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__=="__main__":
    test_health()
    # test_simple_chat()
    # test_chat_with_history()
    # test_chat_with_system_prompt()
    # test_chat_with_options()
    # test_chat_streaming()
    # test_error_missing_message()
    # test_empty_message()
    # test_list_tools()
    # test_tool_calling_simple()
    # test_tool_calling_stream()
    test_chat_with_tools_streaming()
