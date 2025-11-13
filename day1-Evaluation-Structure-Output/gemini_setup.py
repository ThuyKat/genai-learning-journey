from google import genai
from google.genai import types
import os
from google.api_core import retry
from dotenv import load_dotenv
# Load environment variables and API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)


# Automatic retry
def auto_retry():
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

    if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
        genai.models.Models.generate_content = retry.Retry(
            predicate=is_retriable)(genai.models.Models.generate_content)

# Export the client for use in other scripts