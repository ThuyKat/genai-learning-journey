
# Gemini API Python Labs — Learning Summary

## What I Learned & Practiced

**1. Setting Up Gemini API in Python**
[Go to file](./gemini_setup.py)
- Installed and configured `google-generativeai` and `python-dotenv`.
- Secured the API key in a `.env` file.
- Centralized Gemini client setup in `gemini_setup.py` for easy reuse.
- Addressed Gemini API quota errors (`RESOURCE_EXHAUSTED`, 429).
- Added request timing and retry logic for reliable script execution within free tier quotas.
---

**2. Running Prompts and Getting Responses**
[Go to file](./basic-prompt.py)
- Sent prompts to Gemini API using `generate_content`.
- Printed and saved responses for analysis.

---

**3. Multi-turn Conversational Context**
[Go to file](./chat-example.py)
- Explored Gemini’s chat functionality for multi-turn Q&A.
- Observed effects of maintaining conversational history.

---

**4. Exploring Model Variants**
[Code](./thinking-mode.py)
- Listed and inspected Gemini model variants like `gemini-2.0-flash` and `gemini-2.0-flash-thinking-exp`.
- Checked model limitations and configuration options.

---

**5. Response Controls and Sampling**
[Max output tokens](./token-limit.py)
[Top P](./top-p-demo.py)
[Temperature](./temperature-demo.py)
- Used max token count (`max_output_tokens`), temperature, and top-p (`top_p`) to study their impact on creativity and output length.

---

**6. Markdown Transformation/Formatting**
[See code](./execute-code)
-  Output formating and converting: Code outputs were saved in `.json` (for data analysis) and formatted in `.md` (for optimal readability).Improved Gemini API output structure by grouping reasoning, code, and result fields into a single object (`result`). 
- Designed scripts to convert output objects (with reasoning, code, result) into readable Markdown blocks. Created a modular `append_to_md_file()` Python function, along with recursive scripts to convert `.json` outputs (with reasoning, code, and result fields) into clear, well-structured Markdown blocks. 
- Enabled recursive formatting for nested results and code outputs.This handle nested dictionaries and automatically increase Markdown header depth (`##`, `###`, etc.) for each level, resulting in readable and organized reports for both flat and multi-layered data.

---


**7. File-based Context Chaining**
[Chain-of-Thoughts](./CoT.py)
- Developed workflows for reading previous results from logs and appending context or questions, reducing total requests.
- Experimented with chaining reasoning in local files to optimize quota usage.

---
**8. Basic Prompting Techniques**
[Zero shot](./zero-shot.py)
[One and few shot](./one-and-few-shot.py)
[Though,Action,Observation](./ReAct.py)

---

**9. Lab Scripts Created**

- `gemini_setup.py` — shared setup and Gemini client initialization for all experiments
- `basic_prompt.py` — single prompt and response demonstration
- `chat_example.py` — multi-turn conversation script
- `token_limit.py` — output length management (token controls)
- `temperature_demo.py` — output randomness experiments
- `top_p_demo.py` — experiments with top-p (nucleus) sampling settings
- `CoT.py` — chain-of-thought prompting, stepwise Markdown logging
- `execute_code.py` — write and execute code, outputting both .md and .json files, recursive function to convert .json to .md
- `zero_shot.py` — immediate zero-shot inference examples
- `one_and_few_shot.py` — one-shot and few-shot prompting examples
- `thinking_mode.py` — uses flash-thinking-exp model; note: this model is not available on free tier and may hit resource limitations
- `ReAct.py` - solve a question answering task with interleaving Thought, Action, Observation steps

---

## Example Code Snippet

```py
def append_to_md_file(text, md_path):
    with open(md_path, "a") as md_file:
        md_file.write(text)
```

## Notes

- Clarified the difference between Jupyter notebook shell syntax (`!curl`) and Python script download methods (`requests.get`) for efficiently retrieving files from URLs.  
- Addressed common Python errors (like `dict_keys` not subscriptable, and differences between reading a whole JSON vs. JSON Lines format)



