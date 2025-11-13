# import io
# from gemini_setup import client
# import os
# from google.genai import types
# short_config = types.GenerateContentConfig(max_output_tokens=200)
# response = client.models.generate_content_stream(
#     model='gemini-2.0-flash-thinking-exp',
#     contents='Who was the youngest author listed on the transformers NLP paper?',
#     config=short_config
# )

# buf = io.StringIO()
# for chunk in response:
#     buf.write(chunk.text)
#     # Display the response as it is streamed
#     print(chunk.text, end='')

# # And then render the finished response as formatted markdown.
# clear_output()
# os.makedirs("out", exist_ok=True)
# with open("out/thinkmode.md", "w") as f:
#     f.write(buf.getvalue())

# print("Result written to thinkmode.md")

import os
from gemini_setup import client
from google.genai.errors import ClientError

# Prepare the output directory and thought file (append mode)
os.makedirs("out", exist_ok=True)
output_path = "out/thought.md"

def append_to_file(text):
    with open(output_path, "a") as f:
        f.write(text + "\n\n---\n\n")

# Step 1: Initial prompt
initial_prompt = "Who was the youngest author listed on the transformers NLP paper?Lets think step by step"
try:
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=initial_prompt
    )
    initial_thought = response.text
    print(f"Step 1 response:\n{initial_thought}\n{'-'*30}")
    append_to_file(f"**Prompt:** {initial_prompt}\n\n**Response:**\n{initial_thought}")
except ClientError as e:
    print("API error:", e)
    exit(1)

# Step 2: Add another question in context and get a new response
# Read the current context from the file
with open(output_path, "r") as f:
    saved_context = f.read()

next_question = "Based on your previous answer, check your thought process, and do more research"
context_for_next = f"{saved_context}\n\n{next_question}"

try:
    response2 = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=context_for_next
    )
    second_thought = response2.text
    print(f"\nNext response:\n{second_thought}\n{'-'*30}")
    append_to_file(f"**Prompt:** {next_question}\n\n**Response:**\n{second_thought}")
except ClientError as e:
    print("API error:", e)
    exit(1)

print(f"\nAll reasoning is logged in {output_path}")
