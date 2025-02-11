from json import load
import numpy as np
import sys
import re
import os


def get_task(prompt: str) -> str:
    goal_re = re.compile(r"(?<=\n\n)Goal: (?!\w*Goal: ).+", re.MULTILINE + re.DOTALL)
    return goal_re.search(prompt).group().strip() + "\nCOMMANDER response:"


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

commander_das = [
    ("Instruction", 11019, .994),
    ("InformationOnObjectDetails", 6946, .994),
    ("InformationOther", 1148, .8876),
    ("FeedbackPositive", 2745, .9712),
    ("FeedbackNegative", 46, .9565),
    ("Affirm", 460, .7826),
    ("Deny", 161, .7292),
    ("Greetings/Salutations", 2565, .4401),
    ("MiscOther", 607, .5222),
    ("OtherInterfaceComment", 486, .6009),
    ('Acknowledge', 7421, .2138),
    ('AlternateQuestions', 127, .2765),
    ('Confirm', 726, .2575),
    ('NotifyFailure', 408, .0368),
    ('RequestForInstruction', 4043, .007),
    ('RequestForObjectLocationAndOtherDetails', 2010, 0.003),
    ('RequestMore', 503, .002),
    ('RequestOtherInfo', 675, .0075),
]


def das_index(response) -> int:
    if response in das_options:
        return das_options.index(response)
    return len(das_options) - 1


def das_confusion(results_filename="gpt4_result.json", prev=None) -> np.array:

    confusion_matrix = np.zeros((len(das_options), len(das_options)), dtype=int)

    results = load(open(results_filename, "r"))

    for result in results:
        if prev is not None and get_last_turn_type(result["prompt"]) != prev:
            continue
        answer, response = result["answer"], result["response"]
        confusion_matrix[das_index(answer), das_index(response)] += 1

    return confusion_matrix.astype(int)


def das_stats(das_matrix: np.array):
    total_count = 0
    total_score = 0
    for das, count, freq in commander_das:
        i = das_options.index(das)
        tp = das_matrix[i, i]
        fp = das_matrix[i, 1:].sum() - tp
        fn = das_matrix[1:, i].sum() - tp

        fscore = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0

        total_score += fscore * (count * freq)
        total_count += count * freq

        print(f"{das} fscore: {fscore:.2%}")
    print(f"Average fscore: {total_score / total_count:.2%}")


def baseline_das(filename="gpt4_result.json", prev=None) -> np.array:
    confusion_matrix = np.zeros((len(das_options), len(das_options)), dtype=int)

    results = load(open(filename, "r"))

    for result in results:
        prevtype = get_last_turn_type(result["prompt"])
        if prev is not None and prevtype != prev:
            continue
        answer = result["answer"]
        if prevtype == "speak":
            confusion_matrix[das_index(answer), das_index("Instruction")] += 1  # Instruction is majority class
        else:
            confusion_matrix[das_index(answer), das_index("OBSERVE")] += 1

    return confusion_matrix.astype(int)


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


# baseline is when the user speaks only when spoken to by the robot
def calc_baseline(results_filename="gpt4_result.json", prev=None, matrix=False) -> tuple[int, int, int, int] or np.array:
    true_observed = 0
    false_observed = 0
    true_speak = 0
    false_speak = 0

    results = load(open(results_filename, "r"))

    for result in results:
        prevtype = get_last_turn_type(result["prompt"])
        if prev is not None and prevtype != prev:
            continue
        answer = result["answer"].splitlines()[0].strip()
        if answer == "OBSERVE":
            if prevtype != "speak":
                true_observed += 1
            else:
                false_speak += 1
        else:
            if prevtype != "speak":
                false_observed += 1
            else:
                true_speak += 1

    if matrix:
        return np.array([[true_observed, false_speak], [false_observed, true_speak]])
    return true_observed, false_observed, true_speak, false_speak


def conf_matrix(filename="gpt4_result.json", prev=None) -> np.array:
    true_observed, false_observed, true_speak, false_speak, _, _ = calc_score(filename, prev)
    return np.array([[true_observed, false_speak], [false_observed, true_speak]])


def all_matrix(filename="gpt4_result.json", prev=None) -> np.array:
    true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak = calc_score(filename, prev)
    return np.array([[true_observed, false_speak], [false_observed, true_speak], [correct_speak, incorrect_speak]])


def metric_string(filename="gpt4_result.json") -> str:
    return report(*calc_score(filename))


def report(to, fo, ts, fs, cs, is_) -> str:
    to_ret = ""
    to_ret += f"Observed Accuracy: {to / (to + fs):.2%}\n"
    to_ret += f"Speak Accuracy: {ts / (ts + fo):.2%}\n"
    if (is_ + cs) != 0:
        to_ret += f"Dialogue Act Accuracy: {cs / (is_ + cs):.2%}\n\n"

    to_ret += f"Spoke when shouldn't: {fs / (to + fs):.2%}\n"
    to_ret += f"Observed when shouldn't: {fo / (ts + fo):.2%}\n\n"

    to_ret += f"Overall Accuracy: {(to + ts) / (to + fo + ts + fs):.2%}\n"
    to_ret += f"Confusion Matrix:\n"
    to_ret += str(np.array([[to, fs], [fo, ts]]))
    to_ret += "\n"

    to_ret += f"F-Score: {2 * ts / (2 * ts + fs + fo):.2%}"

    return to_ret


def calc_folder(folder_path:str) -> dict:
    true_observed = false_observed = true_speak = false_speak = correct_speak = incorrect_speak = 0
    for filename in os.listdir(folder_path):
        if ".json" not in filename:
            continue
        to, fo, ts, fs, cs, ic = calc_score(os.path.join(folder_path, filename))
        true_observed += to
        false_observed += fo
        true_speak += ts
        false_speak += fs
        correct_speak += cs
        incorrect_speak += ic

    return {
        "f-score": 2 * true_speak / (2 * true_speak + false_speak + false_observed),
        "confusion": np.array([[true_observed, false_speak], [false_observed, true_speak]]),
        "speak_acc": correct_speak / (correct_speak + incorrect_speak),
        "string": report(true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak)
    }


if __name__ == "__main__":
    folder_to_eval = sys.argv[1]
    print(calc_folder(folder_to_eval)["string"])
