from evaluation import summarise_doc,extracted_text
from evaluator import eval_summary
import os

os.makedirs("out",exist_ok=True)

def append_to_file(text,path="out/re-summary.md"):
    with open(path,"a") as f:
        f.write(text+"\n\n---\n\n")



new_prompt = "Explain like I'm 5 the training process"
# Try:
#  ELI5 the training process
#  Summarise the needle/haystack evaluation technique in 1 line
#  Describe the model architecture to someone with a civil engineering degree
#  What is the best LLM?
# Try the following tweaks and see how they positively or negatively change the result:
    #Be specific with the size of the summary,
    #Request specific information,
    #Ask about information that is not in the document,
    #Ask for different degrees of summarisation (such as "explain like I'm 5" or "with full technical depth")

if not new_prompt:
  raise ValueError("Try setting a new summarisation prompt.")


def run_and_eval_summary(prompt):
  """Generate and evaluate the summary using the new prompt."""
  summary = summarise_doc(new_prompt)
  append_to_file(summary)

  text, struct = eval_summary([new_prompt, extracted_text], summary)
  append_to_file(text,path="out/re-evaluation.md")
  print(struct)

run_and_eval_summary(new_prompt)