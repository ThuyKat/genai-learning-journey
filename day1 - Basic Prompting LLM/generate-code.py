from gemini_setup import client
import os
from google.genai import types
from google.genai.errors import ClientError

# Prepare the output directory and thought file (append mode)
os.makedirs("out", exist_ok=True)
output_path = "out/code.md"

def append_to_file(text):
    with open(output_path, "a") as f:
        f.write(text + "\n\n---\n\n")
# The Gemini models love to talk, so it helps to specify they stick to the code if that
# is all that you want.
code_prompt = """
Write a Python function to calculate the factorial of a number. No explanation, provide only the code.
"""
try:
    response = client.models.generate_content(
    model='gemini-2.0-flash',
    config=types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=1024,
    ),
    contents=code_prompt)
    initial_thought = response.text
    print(f"Step 1 response:\n{initial_thought}\n{'-'*30}")
    append_to_file(f"**Prompt:** {code_prompt}\n\n**Response:**\n{initial_thought}")
except ClientError as e:
    print("API error:", e)
    exit(1)