import os
import time
from google import genai
from dotenv import load_dotenv

# Define the model constant for centralized configuration
CURRENT_MODEL = "gemini-3-flash-preview"

def setup_ai(test_connection=True):
    """
    Initializes the AI Client with automatic retry logic for 429 rate limit errors.
    
    Args:
        test_connection (bool): If True, sends a test request to the API.
                                If False, initializes the client without verifying connectivity.
                                Defaults to True.
    Returns:
        genai.Client: The initialized client object, or None if initialization fails.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    # 1. Validate API Key presence
    if not api_key:
        print("[ERROR] No GEMINI_API_KEY found in environment variables.")
        return None
    
    # 2. Security Audit: Check for .gitignore
    # Determines the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    gitignore_path = os.path.join(project_root, ".gitignore")

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            content = f.read()
            if ".env" not in content:
                print("[SECURITY WARNING] .env is NOT listed in .gitignore.")
            else:
                print("[INFO] Git protection active (.env is hidden).")
    else:
        print("[WARNING] No .gitignore file found.")

    # 3. Initialize Client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[ERROR] Client initialization failed: {e}")
        return None

    # Return immediately if testing is not required to conserve quota
    if not test_connection:
        return client

    # 4. Connection Test with Exponential Backoff
    print(f"[INFO] Testing connection to {CURRENT_MODEL}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client.models.generate_content(
                model=CURRENT_MODEL, 
                contents='Ping'
            )
            print(f"[SUCCESS] Connected to {CURRENT_MODEL}")
            return client

        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle Rate Limit (429) errors
            if "429" in error_msg:
                wait_time = (attempt + 1) * 5
                print(f"[WARNING] Rate limit hit (429). Waiting {wait_time}s before retry {attempt+1}...")
                time.sleep(wait_time)
            else:
                # Handle non-retriable errors
                print(f"[ERROR] Connection failed: {e}")
                return None
    
    print("[ERROR] Connection failed after maximum retries.")
    return None