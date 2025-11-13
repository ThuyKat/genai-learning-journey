import os
from google.genai import types
from gemini_setup import client
from evaluation import extracted_text
import functools
os.makedirs("out", exist_ok=True)
output_path = "out/question-set.md"
def append_to_file(text):
    with open(output_path, "a") as f:
        f.write(text + "\n\n---\n\n")



# Try these instructions, or edit and add your own.
terse_guidance = "Answer the following question in a single sentence, or as close to that as possible."
moderate_guidance = "Provide a brief answer to the following question, use a citation if necessary, but only enough to answer the question."
cited_guidance = "Provide a thorough, detailed answer to the following question, citing the document and supplying additional background information as much as possible."
guidance_options = {
    'Terse': terse_guidance,
    'Moderate': moderate_guidance,
    'Cited': cited_guidance,
}

questions = [
    # Un-comment one or more questions to try here, or add your own.
    # Evaluating more questions will take more time, but produces results
    # with higher confidence. In a production system, you may have hundreds
    # of questions to evaluate a complex system.

     "What metric(s) are used to evaluate long context performance?",
    "How does the model perform on code tasks?",
    "How many layers does it have?",
     "Why is it called Gemini?",
]

if not questions:
  raise NotImplementedError('Add some questions to evaluate!')


@functools.cache
def answer_question(question: str, guidance: str = '') -> str:
  """Generate an answer to the question using the uploaded document and guidance."""
  config = types.GenerateContentConfig(
      temperature=0.0,
      system_instruction=guidance,
  )
  response = client.models.generate_content(
      model='gemini-2.0-flash',
      config=config,
      contents=[question, extracted_text],
  )

  return response.text


answer = answer_question(questions[0], terse_guidance)
append_to_file(answer)