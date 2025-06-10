# test_llm.py - Create this as a separate file to test your LLM setup
import os
from dotenv import load_dotenv

print("=== Testing LLM Setup ===")

# Load environment variables
load_dotenv()
print(f"✅ Environment loaded")
print(f"🔑 GROQ_API_KEY exists: {'GROQ_API_KEY' in os.environ}")
if 'GROQ_API_KEY' in os.environ:
    key = os.environ['GROQ_API_KEY']
    print(f"🔑 API Key starts with: {key[:10]}...")

# Test LLM import
try:
    from ourllm import llm

    print("✅ Successfully imported LLM")

    # Test LLM call
    test_message = "Hello, please respond with 'LLM is working correctly'"
    print(f"🧪 Testing with message: {test_message}")

    response = llm.invoke(test_message)
    print(f"✅ LLM Response: {response.content}")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ LLM call error: {e}")

print("=== Test Complete ===")