# Gemini API Python Labs — Learning Summary

## Day 1 Learning Highlight
### Gemini API Document Summarization Workflow
[Link](./evaluation.py)
- Explored end-to-end automation for summarizing long technical documents with Gemini API, using Python scripts.
- Learned to download documents (PDF) programmatically with `requests` or `curl`.
- Understood the limitations of direct PDF upload in Gemini API; extracted document text locally using packages like `pdfplumber`.
- Designed prompts to summarize or analyze content using extracted text as context to stay within quota/token limits.
- Implemented structured Markdown logging to capture API results and summarize outputs in a reusable workflow.
- Improved error handling by adding automatic retry logic and pacing requests to avoid quota-related failures (429).

---

### Document QA: Evaluation Prompt Engineering Workflow
[Make evaluation better or worse](./re-prompt.py)
[Evaluator](./evaluator.py)
- Practiced prompt engineering by experimenting with diverse summary formats:
    - "Explain like I'm 5" (ELI5): Requesting very simple, child-friendly explanations.
    - "Summarize in 1 line": Directing the model to return an extremely concise response.
    - "Describe for a civil engineer": Adapting technical detail and terminology for a specific audience.
    - "Best LLM?": Testing model responses on factual and non-factual queries.
    - Variation in summary length, specificity, and requests for information not present—all to assess the effect on result quality.
- Used evaluation scripts to provide feedback on summary accuracy and structure, storing both raw and evaluated outputs for reproducible documentation.

---

### Answer Questions Based on Uploaded Document
[Link to code](./question_set.py)
- Evaluation framework design: learned to define multiple guidance modes (terse, moderate, cited) to benchmark the same questions under varying depth levels—key for model evaluation experiments.
- Automated Q&A pipelines:  built a basic pipeline that loads extracted text (extracted_text from evaluation), runs model queries, and stores structured answers—forming the core of an evaluation harness for AI documentation or RAG systems.
- Caching optimization: Used functools.cache to save API cost and runtime when re-evaluating identical questions.

---

### Assess Quality of Answers
[Link](./eval_answer.py)
- Extended pipeline to automatically evaluate AI answers against a rubric by prompting the model as an expert grader and coercing the final score into a strict enum type.​
- Built an evaluator prompt that defines task, metrics (instruction following, groundedness, completeness, fluency), rubric, and stepwise scoring, producing both an explanatory rationale and a numeric score.
- Created a two-stage chat: first, generate a verbose evaluation using a grading prompt; second, request a strictly typed final score using a response schema.

- Structured scoring with enums: Used a generation config with response_mime_type set to text/x.enum and response_schema bound to an Enum class to force the model to return a single value among {5,4,3,2,1}. This approach reduces post-processing complexity and parsing errors, since the SDK parses to response.parsed directly from the enum schema.​

- Added functools.cache to memoize eval_answer so repeated evaluations for the same (prompt, response) pair avoid extra API calls and ensure consistent outputs across runs.
- Appended the verbose textual evaluation to out/eval-answer.md with separators, enabling auditability and later manual review of rationales alongside the machine-readable score.

---

### Answer Under Each Type of Guidance
[Link](./question_set.py)
- Set up a parameterized Q&A generator that applies different guidance styles, caches answers per question+style, and logs outputs for later evaluation or audits.
- Defined three guidance modes (Terse, Moderate, Cited) and passed them as system_instruction via GenerateContentConfig to shape tone, length, and citation behavior deterministically with temperature 0.0.
- Combined the user question and an extracted_text context in contents so the model answers from provided material rather than external knowledge, aligning with doc-grounded Q&A setups.
- Wrapped answer_question with functools.cache to memoize by (question, guidance), reducing duplicate API calls and ensuring repeatable outputs for the same inputs.

---

### Assess Quality of Guidance Prompts
[Link](./eval_loop.py)
- Built a small evaluation harness that generates answers under different guidance styles, scores them with a rubric-driven LLM evaluator, aggregates scores, and reports per-guidance averages with nearest enum labels.
- Multi-pass evaluation loop: Iterates over each question and each guidance mode to produce multiple answers per configuration, enabling error reduction and stability checks via NUM_ITERATIONS.
- Testing multiple guidance prompts (Terse, Moderate, Cited) lets you quantify instruction tuning effects on instruction following, groundedness, completeness, and fluency across a set of prompts.
- Accumulates scores per guidance style in a defaultdict, then computes average scores over all questions and iterations to compare guidance effectiveness quantitatively.
- Maps the floating-point average to the nearest discrete rating by rounding and casting back to the AnswerRating enum for a readable summary label.

---

### Compare 2 Types of Guidance
[Link](./pairwise.py)
- A pairwise evaluator that compares two model responses to the same prompt of different guidance styles
- Asks a separate “judge” model to read the question plus both answers and explain which one is better and why. The judge then outputs a simple label: A (terse wins), SAME (tie), or B (cited wins), along with the explanation text.
- Pairwise judging mirrors how humans compare options: it’s often easier and more reliable to pick a better answer between two than to assign absolute scores.
- Running this across many prompts lets you see which guidance style generally wins, so you can rank prompts or choose defaults.

---

### Create Comparator to Sort Prompts
[Link](./comparator.py)
- Turned each guidance style into a comparable “object” that can be sorted by how often it wins in pairwise head‑to‑head tests across your questions.
- Wraps a guidance prompt in a small class with compare methods, so two prompts can be directly compared and even sorted from best to worst.
- For any two prompts, it generates answers to the same question, asks the judge to pick A/SAME/B, converts that to +1/0/−1, repeats over n trials and all questions, then averages to get who’s better overall.
- "eq" controls “are these two things equal?” when you use ==, letting you decide what equality means for your class."lt" controls “is this less than that?” when you use <, letting you define your own ordering rule.

---
