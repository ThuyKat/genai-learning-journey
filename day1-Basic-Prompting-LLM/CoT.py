from gemini_setup import client
import os

prompt = """When I was 4 years old, my partner was 3 times my age. Now, I
am 20 years old. How old is my partner? Return the answer directly."""

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt)

print(response.text)

# think step by step
prompt = """When I was 4 years old, my partner was 3 times my age. Now,
I am 20 years old. How old is my partner? Let's think step by step."""

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt)

# After receiving the response
    # Ensure the output folder exists
os.makedirs("out", exist_ok=True)
with open("out/result.md", "w") as f:
    f.write(response.text)

print("Result written to result.md")
