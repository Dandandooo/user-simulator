from typing import Generator
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = [
    "OBSERVE", "Acknowledge", "Affirm", "AlternateQuestions",
    "Confirm", "Deny", "FeedbackNegative", "FeedbackPositive",
    "Greetings/Salutations", "InformationOnObjectDetails",
    "InformationOther", "Instruction", "MiscOther", "NotifyFailure",
    "OtherInterfaceComment", "RequestForInstruction",
    "RequestForObjectLocationAndOtherDetails", "RequestMore",
    "RequestOtherInfo", "OTHER"
]

# TODO: Visualize model accuracy
# TODO: Heatmap of confusion matrix
class Evaluate:
    def __init__(self, path: str):
        self.path = path

        self.confusion_matrix = np.zeros((len(LABELS), len(LABELS)))

        for response, truth in self._scan_file(path):
            response_label = self._response_to_label(response)

            self.confusion_matrix[self._position(response_label, truth)] += 1

        self.confusion_matrix = self.confusion_matrix.astype(int)
        self.stats = self.matrix_metrics(self.confusion_matrix)


    @staticmethod
    def _response_to_label(response: str) -> str:
        lowered = response.strip().lower()
        for i, label in enumerate(LABELS[:-1]):
            if lowered.startswith(label.lower()) or lowered.endswith(label.lower()):
                return label
        return "OTHER"

    @staticmethod
    def _position(label: str, truth: str) -> int:
        return LABELS.index(truth), LABELS.index(label)

    @staticmethod
    def matrix_metrics(matrix: np.ndarray):
        """
        First column and row is Observe
        Last column and row is Other
        """
        speak_matrix = np.array([
            [matrix[0][0], np.sum(matrix[0][1:])],
            [np.sum(matrix[1:, 0]), np.sum(matrix[1:, 1:])]
        ])

        speak_f1 = 2 * speak_matrix[1, 1] / (2 * speak_matrix[1, 1] + speak_matrix[0, 1] + speak_matrix[1, 0])

        # d, e
        da_matrix = matrix[1:, 1:]
        da_frequencies = np.sum(da_matrix, axis=1).flatten()
        da_f1s = np.array([2 * da_matrix[i, i] / (np.sum(da_matrix[i, :]) + np.sum(da_matrix[:, i])) for i in range(da_matrix.shape[0])])
        da_f1 = np.average(da_f1s, weights=da_frequencies)

        # c, d, e
        spandana_matrix = matrix
        spandana_frequencies = np.sum(spandana_matrix, axis=1).flatten()[1:-1]
        spandana_f1s = np.array([2 * spandana_matrix[i, i] / (np.sum(spandana_matrix[i, :]) + np.sum(spandana_matrix[:, i])) for i in range(1, spandana_matrix.shape[0]-1)])
        spandana_accuracies = np.array([spandana_matrix[i, i] / np.sum(spandana_matrix[i, :]) for i in range(1, spandana_matrix.shape[0]-1)])
        spandana_f1 = np.average(spandana_f1s, weights=spandana_frequencies)
        spandana_accuracy = np.average(spandana_accuracies, weights=spandana_frequencies)

        return {
            "speak_f1": speak_f1,
            "speak_matrix": speak_matrix,
            "da_f1": da_f1,
            "cde_f1": spandana_f1,
            "cde_acc": spandana_accuracy,
        }

    @staticmethod
    def _scan_file(filename: str) -> Generator[tuple[str, str], None, None]:
        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            entry = json.loads(line)
            # I messed up the format of the file, so I have to do this
            response = entry["response"]
            while isinstance(response, dict):
                response = response["response"]
            if not isinstance(response, str):
                print(type(response))
                continue
            yield response, entry["truth"]

    def print_results(self):
        print("\x1b[35;1mConfusion Matrix:\x1b[0m")
        print(self.confusion_matrix, end="\n\n")

        print("\x1b[35;1mMetrics:\x1b[0m\n")

        print("\x1b[33mSpeak F1:\x1b[0m", self.stats["speak_f1"])
        print("\x1b[33mSpeak Matrix:\x1b[0m")
        print(self.stats["speak_matrix"], end="\n\n")

        print("\x1b[33mDA F1:\x1b[0m", self.stats["da_f1"])

        print("\x1b[33mCDE F1:\x1b[0m", self.stats["cde_f1"])
        print("\x1b[33mCDE Accuracy:\x1b[0m", self.stats["cde_acc"])


    def heatmap(self):
        sns.heatmap(self.confusion_matrix, annot=True, xticklabels=LABELS, yticklabels=LABELS)


# The input is a dictionary with keys being the model name, and values being a dictionary of experiments paired with Evaluates
def graph_comparison(evaluations: dict[str, dict[str, Evaluate]]):
    raise NotImplementedError()
