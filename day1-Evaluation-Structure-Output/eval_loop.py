import collections
import itertools
from question_set import questions, answer, guidance_options
from eval_answer import AnswerRating, eval_answer
from question_set import answer_question

# Number of times to repeat each task in order to reduce error and calculate an average.
# Increasing it will take longer but give better results, try 2 or 3 to start.
NUM_ITERATIONS = 1

scores = collections.defaultdict(int)
responses = collections.defaultdict(list)

for question in questions:
  print(question+":")
  for guidance, guide_prompt in guidance_options.items():

    for n in range(NUM_ITERATIONS):
      # Generate a response.
      answer = answer_question(question, guide_prompt)

      # Evaluate the response (note that the guidance prompt is not passed).
      written_eval, struct_eval = eval_answer(question, answer, n)
      print(f'{guidance}: {struct_eval}')

      # Save the numeric score.
      scores[guidance] += int(struct_eval.value)

      # Save the responses, in case you wish to inspect them.
      responses[(guidance, question)].append((answer, written_eval))
for guidance, score in scores.items():
  avg_score = score / (NUM_ITERATIONS * len(questions))
  nearest = AnswerRating(str(round(avg_score)))
  print(f'{guidance}: {avg_score:.2f} - {nearest.name}')