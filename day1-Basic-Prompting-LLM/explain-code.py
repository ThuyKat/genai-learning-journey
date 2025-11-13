from gemini_setup import client
import os
import requests

os.makedirs("out",exist_ok=True)
output_path="out/explain.md"
def append_to_file(text):
    with open(output_path, "a") as file:
        file.write(text)

url = "https://raw.githubusercontent.com/magicmonty/bash-git-prompt/refs/heads/master/gitprompt.sh"
response = requests.get(url)
explain_prompt = f"""
Please explain what this file does at a very high level. What is it, and why would I use it?

```
{url}
```
"""

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=explain_prompt)

append_to_file(response.text)