from question_set import questions,answer_question,terse_guidance,moderate_guidance,cited_guidance
import functools
from eval_loop import NUM_ITERATIONS
from pairwise import eval_pairwise, AnswerComparison
from gemini_setup import auto_retry
@functools.total_ordering
class QAGuidancePrompt:
  """A question-answering guidance prompt or system instruction."""

  def __init__(self, prompt, questions, n_comparisons=NUM_ITERATIONS):
    """Create the prompt. Provide questions to evaluate against, and number of evals to perform."""
    self.prompt = prompt
    self.questions = questions
    self.n = n_comparisons

  def __str__(self):
    return self.prompt

  def _compare_all(self, other):
    """Compare two prompts on all questions over n trials."""
    results = [self._compare_n(other, q) for q in questions]
    mean = sum(results) / len(results)
    return round(mean)

  def _compare_n(self, other, question):
    """Compare two prompts on a question over n trials."""
    results = [self._compare(other, question, n) for n in range(self.n)]
    mean = sum(results) / len(results)
    return mean

  def _compare(self, other, question, n=1):
    """Compare two prompts on a single question."""
    answer_a = answer_question(question, self.prompt)
    answer_b = answer_question(question, other.prompt)

    _, result = eval_pairwise(
        prompt=question,
        response_a=answer_a,
        response_b=answer_b,
        n=n,  # Cache buster
    )
    # print(f'q[{question}], a[{self.prompt[:20]}...], b[{other.prompt[:20]}...]: {result}')

    # Convert the enum to the standard Python numeric comparison values.
    if result is AnswerComparison.A:
      return 1
    elif result is AnswerComparison.B:
      return -1
    else:
      return 0

  def __eq__(self, other):
    """Equality check that performs pairwise evaluation."""
    if not isinstance(other, QAGuidancePrompt):
      return NotImplemented

    return self._compare_all(other) == 0

  def __lt__(self, other):
    """Ordering check that performs pairwise evaluation."""
    if not isinstance(other, QAGuidancePrompt):
      return NotImplemented

    return self._compare_all(other) < 0

auto_retry()
terse_prompt = QAGuidancePrompt(terse_guidance, questions)
moderate_prompt = QAGuidancePrompt(moderate_guidance, questions)
cited_prompt = QAGuidancePrompt(cited_guidance, questions)

# Sort in reverse order, so that best is first
sorted_results = sorted([terse_prompt, moderate_prompt, cited_prompt], reverse=True)
for i, p in enumerate(sorted_results):
  if i:
    print('---')

  print(f'#{i+1}: {p}')