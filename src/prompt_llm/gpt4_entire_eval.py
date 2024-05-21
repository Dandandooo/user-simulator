from json import load
import numpy as np
import sys
import re


# Returns the type of the last robot turn
def get_last_turn_type(prompt: str) -> str or None:
    goal_re = re.compile(r"(?<=\n\n\n)Goal: (?:(?!COMMANDER response:).)+", re.MULTILINE + re.DOTALL)
    turns = goal_re.search(prompt).group().strip()
    final_re = re.compile(r"(?<=DRIVER:)(?!.*DRIVER:)(?P<action>.+)", re.MULTILINE + re.DOTALL)
    final_turn = final_re.search(turns)
    if final_turn is not None:
        final_turn = final_turn.group("action").strip()
    else:
        return None
    if re.match(r"<.*>", final_turn):
        return "observe" if final_turn == "<observe>" else "action"
    else:
        return "speak"


das_options = [
        "OBSERVE",
        'Acknowledge',
        'Affirm',
        'AlternateQuestions',
        'Confirm',
        'Deny',
        'FeedbackNegative',
        'FeedbackPositive',
        'Greetings/Salutations',
        'InformationOnObjectDetails',
        'InformationOther',
        'Instruction',
        'MiscOther',
        'NotifyFailure',
        'OtherInterfaceComment',
        'RequestForInstruction',
        'RequestForObjectLocationAndOtherDetails',
        'RequestMore',
        'RequestOtherInfo',
        "OTHER"
    ]


def das_index(response) -> int:
    if response in das_options:
        return das_options.index(response)
    return len(das_options) - 1


def das_confusion(results_filename="gpt4_result.json", prev=None) -> np.array:

    confusion_matrix = np.zeros((len(das_options), len(das_options)))

    results = load(open(results_filename, "r"))

    for result in results:
        if prev is not None and get_last_turn_type(result["prompt"]) != prev:
            continue
        answer, response = result["answer"], result["response"]
        confusion_matrix[das_index(answer), das_index(response)] += 1

    return confusion_matrix


def das_stats(das_matrix: np.array):
    for i, das in enumerate(das_options[:-1]):
        tp = das_matrix[i, i]
        fp = das_matrix[i].sum() - tp
        fn = das_matrix[:, i].sum() - tp

        fscore = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0

        print(f"{das} fscore: {fscore:.2%}")


def calc_score(results_filename="gpt4_result.json", prev=None) -> tuple[int, int, int, int, int, int]:
    true_observed = 0
    false_observed = 0
    true_speak = 0
    false_speak = 0
    correct_speak = 0
    incorrect_speak = 0

    results = load(open(results_filename, "r"))

    for result in results:
        if prev is not None and get_last_turn_type(result["prompt"]) != prev:
            continue
        answer, response = result["answer"], result["response"]
        if answer == "OBSERVE":
            if "OBSERVE" in response.upper():
                true_observed += 1
            else:
                false_speak += 1
        else:
            if "OBSERVE" in response.upper():
                false_observed += 1
            else:
                true_speak += 1
                if response == answer:
                    correct_speak += 1
                else:
                    incorrect_speak += 1

    return true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak


def conf_matrix(filename="gpt4_result.json", prev=None) -> np.array:
    true_observed, false_observed, true_speak, false_speak, _, _ = calc_score(filename, prev)
    return np.array([[true_observed, false_speak], [false_observed, true_speak]])


def all_matrix(filename="gpt4_result.json", prev=None) -> np.array:
    true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak = calc_score(filename, prev)
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
