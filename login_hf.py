import os
from huggingface_hub import login, HfApi

# Security improvement: Use an environment variable for the HF token
token = os.getenv("HF_TOKEN")

if not token:
    print("⚠️  HF_TOKEN environment variable not set. Please set it to your Hugging Face token.")
    print("Example: set HF_TOKEN=your_token_here (on Windows) or export HF_TOKEN=your_token_here (on Linux/Mac)")
else:
    try:
        print("Attempting to log in...")
        login(token=token)
        
        # Verify the login worked
        api = HfApi()
        user = api.whoami()
        print(f"✅ Success! Logged in as: {user['name']}")
        
    except Exception as e:
        print(f"❌ Login failed! Error: {e}")