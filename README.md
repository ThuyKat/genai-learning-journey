# genai-learning-journey

[Link on Kaggle ](https://www.kaggle.com/learn-guide/5-day-genai)


## Day 1A: Gemini API Setup & Basic Prompting

[View scripts and materials →](./day1-Basic-Prompting-LLM)

- Installed and configured the Gemini API Python client (`google-generativeai`), secured API key with `.env`.
- Created `gemini_setup.py` for easy client reuse in all experiments.
- Addressed API quota/rate errors (429) and added retry/backoff logic.
- Sent basic prompts, received responses, and logged results.
- Explored multi-turn chat interactions and conversational context using `chat-example.py`.
- Tried model variants (`gemini-2.0-flash`, `gemini-2.0-flash-thinking-exp`), studied API config options.
- Controlled outputs with max tokens, temperature, and top-p sampling.
- Designed scripts to save results in Markdown and JSON for analysis.
- Practiced basic prompt types: zero-shot, one-shot, few-shot, and ReAct-style workflows with chain-of-thought logging.
- Developed file-based context chaining (reading/writing previous results to optimize context and lower API usage).
- Code output example:
  ```python
  def append_to_md_file(text, md_path):
      with open(md_path, "a") as md_file:
          md_file.write(text)
  ```
- Notes: Clarified difference between Jupyter shell commands, Python request libraries, and common Python data handling errors.

***

## Day 1B: Document Q&A, Evaluation, and Guidance Comparison

[View scripts and materials →](./day1-Evaluation-Structure-Output)

- Explored end-to-end document summarization with Gemini API, including downloading PDFs and extracting text (e.g., with `pdfplumber`).
- Created prompts to summarize/analyze text, designed for context length and token limits.
- Structured all API outputs (answers, grades, comparisons) in Markdown files for reproducibility.
- Practiced prompt engineering: varied summary formats ("Explain like I'm 5", "Summarize in 1 line", audience-specific, and factual checks).
- Automated Q&A pipeline tested multiple guidance styles (terse, moderate, cited), benchmarking effectiveness and logging outputs.
- Automated evaluation: extended pipeline to grade answers by strict rubric (instruction following, groundedness, completeness, fluency) and output numeric enum scores.
- Used functools.cache for answer/evaluation memoization, lowering cost and increasing repeatability.
- Ran multi-pass eval loops over guidance/question pairs, aggregated averages, and mapped to nearest enum scores for leaderboard summaries.
- Performed pairwise "A/B" judging (which guidance wins per question), storing rationale and results.
- Built sortable comparator class for ranking guidance prompts by win rates using custom equality and less-than logic.

