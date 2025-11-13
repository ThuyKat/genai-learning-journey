
# Gemini API Python Labs — Learning Summary

## What I Learned & Practiced

**1. Setting Up Gemini API in Python**
- Installed and configured `google-generativeai` and `python-dotenv`.
- Secured the API key in a `.env` file.
- Centralized Gemini client setup in `gemini_setup.py` for easy reuse.

---

**2. Running Prompts and Getting Responses**
- Sent prompts to Gemini API using `generate_content`.
- Printed and saved responses for analysis.

---

**3. Multi-turn Conversational Context**
- Explored Gemini’s chat functionality for multi-turn Q&A.
- Observed effects of maintaining conversational history.

---

**4. Exploring Model Variants**
- Listed and inspected Gemini model variants like `gemini-2.0-flash` and `gemini-2.0-flash-thinking-exp`.
- Checked model limitations and configuration options.

---

**5. Response Controls and Sampling**
- Used max token count (`max_output_tokens`), temperature, and top-p (`top_p`) to study their impact on creativity and output length.

---

**6. Logging and Markdown Output**
- Built modular Python functions to append reasoning, code, and results to Markdown (`out/thought.md`).
- Automated Markdown documentation for stepwise model output and code logs.

---

**7. Markdown Transformation/Formatting**
- Designed scripts to convert output objects (with reasoning, code, result) into readable Markdown blocks.
- Enabled recursive formatting for nested results and code outputs.

---

**8. Handling API Rate Limits**
- Addressed Gemini API quota errors (`RESOURCE_EXHAUSTED`, 429).
- Added request timing and retry logic for reliable script execution within free tier quotas.

---

**9. File-based Context Chaining**
- Developed workflows for reading previous results from logs and appending context or questions, reducing total requests.
- Experimented with chaining reasoning in local files to optimize quota usage.

---

**10. Lab Scripts Created**

- `gemini_setup.py` — shared setup and Gemini client initialization for all experiments
- `basic_prompt.py` — single prompt and response demonstration
- `chat_example.py` — multi-turn conversation script
- `token_limit.py` — output length management (token controls)
- `temperature_demo.py` — output randomness experiments
- `CoT.py` — chain-of-thought prompting, stepwise Markdown logging
- `execute_code.py` — write and execute code, outputting both .md and .json files
- `one_and_few_shot.py` — one-shot and few-shot prompting examples
- `thinking_mode.py` — uses flash-thinking-exp model; note: this model is not available on free tier and may hit resource limitations
- `top_p_demo.py` — experiments with top-p (nucleus) sampling settings
- `zero_shot.py` — immediate zero-shot inference examples
- Function for recursive Markdown formatting in `execute-code.py`

---

## Example Code Snippet

```py
def append_to_md_file(text, md_path):
    with open(md_path, "a") as md_file:
        md_file.write(text)
```

### Example Output Transformation

```json
{
  "result": {
    "text": "...summary...",
    "executable_code": {"code": "...", "language": "PYTHON"},
    "code_execution_result": {"output": "..."}
  }
}
```

Converted to Markdown:


## Reasoning
...summary...

## Code
```python
...
```

## Execution Result
...

---



