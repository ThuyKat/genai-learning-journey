from gemini_setup import client
from google.genai import types

model_config = types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
)

story_prompt = "You are a creative writer. Write a short story about a cat who goes on an adventure."
response = client.models.generate_content(
    model='gemini-2.0-flash',
    config=model_config,
    contents=story_prompt
)
print(response.text)
