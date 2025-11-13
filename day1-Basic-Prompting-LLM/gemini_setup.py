# gemini_setup.py
from google import genai
from dotenv import load_dotenv
import os

# Load environment variables and API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Export the client for use in other scripts
