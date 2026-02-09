"""
Troubleshoot Gemini API connectivity and model availability.
Tests different model names and reports status.
"""

import os
from time import sleep

# Models to test - various naming conventions Google has used
MODELS_TO_TEST = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro",
]

def test_with_langchain(model_name: str) -> dict:
    """Test model using LangChain ChatGoogleGenerativeAI"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        model = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        response = model.invoke("Say 'hello' and nothing else.")
        return {
            "status": "OK",
            "response": response.content[:100] if response.content else "empty"
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {str(e)[:200]}"
        }


def test_with_google_genai(model_name: str) -> dict:
    """Test model using google.generativeai directly"""
    try:
        import google.genai as genai

        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'hello' and nothing else.")
        return {
            "status": "OK",
            "response": response.text[:100] if response.text else "empty"
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {str(e)[:200]}"
        }


def list_available_models():
    """List all available models from the API"""
    try:
        import google.genai as genai

        print("\n" + "="*60)
        print("AVAILABLE MODELS FROM API")
        print("="*60)

        models = genai.list_models()
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(f"  {model.name}")
        print()
    except Exception as e:
        print(f"Failed to list models: {type(e).__name__}: {e}")


def check_api_key():
    """Check if API key is set"""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        print(f"API Key found: {key[:8]}...{key[-4:]}")
        return True
    else:
        print("WARNING: No GOOGLE_API_KEY or GEMINI_API_KEY found in environment")
        return False


def check_quota_info():
    """Try to get quota/billing info"""
    print("\nQuota troubleshooting tips:")
    print("  - Check https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
    print("  - Verify billing is enabled on your project")
    print("  - Check if you're using the right API key (AI Studio vs Cloud)")
    print("  - Rate limits: Free tier = 60 requests/min, Paid = higher")


def main():
    print("="*60)
    print("GEMINI API TROUBLESHOOTER")
    print("="*60)

    # Check API key
    print("\n[1] Checking API Key...")
    if not check_api_key():
        print("Set your API key with: export GOOGLE_API_KEY='your-key'")
        return

    # List available models
    print("\n[2] Listing available models...")
    list_available_models()

    # Test models with LangChain
    print("\n[3] Testing models with LangChain...")
    print("-"*60)

    for model_name in MODELS_TO_TEST:
        print(f"\nTesting: {model_name}")
        result = test_with_langchain(model_name)
        if result["status"] == "OK":
            print(f"  ✓ OK - Response: {result['response']}")
        else:
            print(f"  ✗ FAILED - {result['error']}")

        # Small delay to avoid rate limiting
        sleep(1)

    # Test with direct SDK
    print("\n" + "-"*60)
    print("\n[4] Testing first model with google.generativeai SDK...")

    try:
        result = test_with_google_genai(MODELS_TO_TEST[0])
        if result["status"] == "OK":
            print(f"  ✓ OK - Response: {result['response']}")
        else:
            print(f"  ✗ FAILED - {result['error']}")
    except ImportError:
        print("  google-generativeai package not installed")

    # Quota info
    print("\n" + "-"*60)
    check_quota_info()

    print("\n" + "="*60)
    print("TROUBLESHOOTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
