from gemini_setup import client
from google.genai import types

high_temp_config = types.GenerateContentConfig(temperature=2.0)
low_temp_config = types.GenerateContentConfig(temperature=0.0)

print("High temperature results:")
for _ in range(5):
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=high_temp_config,
        contents='Pick a random colour... (respond in a single word)'
    )
    if response.text:
        print(response.text, '-' * 25)

print("\nLow temperature results:")
for _ in range(5):
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=low_temp_config,
        contents='Pick a random colour... (respond in a single word)'
    )
    if response.text:
        print(response.text, '-' * 25)
