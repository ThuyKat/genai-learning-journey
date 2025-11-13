from gemini_setup import client
import requests
import os
from google.genai import types
from gemini_setup import auto_retry
import pdfplumber  # Requires: pip install pdfplumber

os.makedirs("out", exist_ok=True)
output_path = "out/summary.md"
def append_to_file(text):
    with open(output_path, "a") as f:
        f.write(text + "\n\n---\n\n")

auto_retry()

# Download the PDF
url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
response = requests.get(url)
with open("gemini.pdf", "wb") as f:
    f.write(response.content)

# Extract text from the PDF
with pdfplumber.open("gemini.pdf") as pdf:
    extracted_text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"

# Truncate for token limit (adjust as appropriate!)
extracted_text = extracted_text[:5000]  # Or further segment for very long docs

request = f"{extracted_text}\n\nTell me about the training process used here."

def summarise_doc(request: str) -> str:
    """Execute the request with low temperature for stable output."""
    config = types.GenerateContentConfig(temperature=0.0)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=config,
        contents=request
    )
    return response.text

summary = summarise_doc(request)
append_to_file(summary)
print("Summary written to", output_path)
