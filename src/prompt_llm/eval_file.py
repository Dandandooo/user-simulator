import sys

from src.prompt_llm.gpt4_entire_eval import calc_score, das_stats, metric_string

filename = sys.argv[1]

score = metric_string(filename)

to, fo, ts, fs, _, _ = calc_score(filename)
fscore = 2 * ts / (2 * ts + fo + fs)

print(score)
print()
print(f"F-Score: {fscore:.2%}")