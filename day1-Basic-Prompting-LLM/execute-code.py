from pprint import pprint
from gemini_setup import client
import os
from google.genai import types
from google.genai.errors import ClientError
import json
# Prepare the output directory and thought file (append mode)
os.makedirs("out", exist_ok=True)
output_path = "out/code-execute.json"
md_path = "out/code-execute.md"
def append_to_file(result_obj):
    with open(output_path, "a") as f:
        json.dump({"result": result_obj}, f)
        f.write("\n")  # newline separates JSON objects
def append_to_md_file(text):
    with open(md_path, "a") as md_file:
        md_file.write(text)
config = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)
def display(part_dict, level=2):
    for k, v in part_dict.items():  # This gives both key and value
        result_obj[k] = v
        heading = "#" * level
        append_to_md_file(f"{heading} {k}\n\n")
        # If value is itself a dict (nested), recurse into it
        if isinstance(v, dict) and len(v) > 0:
            display(v,level+1)
        else:
             append_to_md_file(f"{v}\n\n")
code_exec_prompt = """
Generate the first 14 odd prime numbers, then calculate their sum.
"""
try:
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=config,
        contents=code_exec_prompt)
    result_obj = {}
    for part in response.candidates[0].content.parts:
        pprint(part.to_json_dict())
        print("-----")

        part_dict = part.to_json_dict()
        
        display(part_dict)
    append_to_file(result_obj)
    
except ClientError as e:
    print("API error:", e)
    exit(1)