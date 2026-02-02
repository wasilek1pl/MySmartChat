import os
from google import genai  # Notice the import change
from dotenv import load_dotenv

def setup_ai():
    """
    Updated 2025 Setup: Initializes the new google-genai Client.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    print("\n--- 🛡️ DATA ENGINEER SAFETY AUDIT (v2.0) ---")

    # 1. Key Presence Check
    if not api_key:
        print("❌ ERROR: No API Key found in your .env file.")
        return None

    # 2. Security Check (Git Protection)
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            content = f.read()
            if ".env" not in content:
                print("⚠️ SECURITY WARNING: .env is NOT in .gitignore!")
            else:
                print("🔒 Git Protection: Active (.env is hidden)")
    else:
        print("⚠️ Warning: No .gitignore file found.")

    # 3. Initialize the New Client
    try:
        # The new SDK uses a Client object that holds your key
        client = genai.Client(api_key=api_key)
        
        # Test the connection with a tiny request
        # In the new SDK, we use client.models.generate_content
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents='Ping'
        )
        
        print("✅ Model: Gemini 2.5 Flash (Unified SDK)")
        print("------------------------------------------\n")
        
        return client  # We return the CLIENT now, not the model

    except Exception as e:
        error_msg = str(e).lower()
        if "location not supported" in error_msg:
            print("🌍 REGION ERROR: Please ensure your VPN is set up.")
        elif "429" in error_msg:
            print("🛑 LIMIT REACHED: You hit the free cap. Wait 60s.")
        else:
            print(f"❌ CONNECTION FAILED: {e}")
        return None