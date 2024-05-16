from json import load
import numpy as np
import sys


def calc_score(results_filename="gpt4_result.json") -> tuple[int, int, int, int, int, int]:
    true_observed = 0
    false_observed = 0
    true_speak = 0
    false_speak = 0
    correct_speak = 0
    incorrect_speak = 0

    results = load(open(results_filename, "r"))

    for result in results:
        answer, response = result["answer"], result["response"]
        if answer == "OBSERVE":
            if response == "OBSERVE":
                true_observed += 1
            else:
                false_speak += 1
        else:
            if response == "OBSERVE":
                false_observed += 1
            else:
                true_speak += 1
                if response == answer:
                    correct_speak += 1
                else:
                    incorrect_speak += 1

    return true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak


def conf_matrix(filename="gpt4_result.json") -> np.array:
    true_observed, false_observed, true_speak, false_speak, _, _ = calc_score(filename)
    return np.array([[true_observed, false_speak], [false_observed, true_speak]])


def all_matrix(filename="gpt4_result.json") -> np.array:
    true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak = calc_score(filename)
    return np.array([[true_observed, false_speak], [false_observed, true_speak], [correct_speak, incorrect_speak]])


def metric_string(filename="gpt4_result.json") -> str:
    to_ret = ""
    true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak = calc_score(filename)
    to_ret += f"Observed Accuracy: {true_observed / (true_observed + false_speak):.2%}\n"
    to_ret += f"Speak Accuracy: {true_speak / (true_speak + false_observed):.2%}\n"
    if (incorrect_speak + correct_speak) != 0:
        to_ret += f"Dialogue Act Accuracy: {correct_speak / (incorrect_speak + correct_speak):.2%}\n\n"

    to_ret += f"Spoke when shouldn't: {false_speak / (true_observed + false_speak):.2%}\n"
    to_ret += f"Observed when shouldn't: {false_observed / (true_speak + false_observed):.2%}\n\n"

    to_ret += f"Overall Accuracy: {(true_observed + true_speak) / (true_observed + false_observed + true_speak + false_speak):.2%}\n"
    to_ret += f"Confusion Matrix:\n"
    to_ret += str(np.array([[true_observed, false_speak], [false_observed, true_speak]]))

    return to_ret


if __name__ == "__main__":
    print(metric_string())
